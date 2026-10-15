"""基于 SSIM 的相似图像检索工具（仿 AntiDupl，仅保留 SSIM 比较）。

AntiDupl / AntiDuplX 的核心思路是：把图片统一缩放到一个很小的比较尺寸（默认 32x32），
再在该尺寸下计算结构相似度（SSIM），相似度高于阈值的图片对视为「相似/重复图片」。
本模块只保留这一条 SSIM 比较链路，不包含 AntiDupl 的模糊、损坏图检测等其它算法。

参考参数（AntiDuplX 命令行默认值）：

* ``-cs=32x32`` 比较尺寸（可选 16x16 / 32x32 / 64x64）
* ``-ct=0.05``  比较阈值，等价于相似度 ``1 - 0.05 = 0.95``
* ``-sd=1``     递归搜索子目录

本模块只依赖 ``numpy`` 与 ``Pillow``，不依赖 OpenCV / scikit-image。
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Pillow 用于解码与缩放，缺失时函数会给出明确错误
    from PIL import Image
except ImportError:  # pragma: no cover - 仅在未安装 Pillow 的环境触发
    Image = None


#: 默认比较尺寸（AntiDuplX 默认 32x32）
DEFAULT_COMPARE_SIZE = 32
#: 允许的比较尺寸选项（AntiDuplX 支持 16/32/64，额外提供 128 以便更精细比较）
COMPARE_SIZE_CHOICES = (16, 32, 64, 128)
#: 默认相似度阈值（AntiDuplX compareThreshold=0.05 的等价相似度）
DEFAULT_SIMILARITY_THRESHOLD = 0.95
#: 支持的图片扩展名（对齐 AntiDupl 常见格式）
IMAGE_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".tif", ".tiff", ".webp", ".jfif",
)
#: SSIM 标准窗口尺寸与高斯标准差（论文默认 11x11 / sigma=1.5）
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5
#: 每次批量比较的图片数量上限，用于控制内存峰值
_COMPARE_BATCH_PIXELS = 2_000_000

#: SSIM 稳定性常数（动态范围 L = 1.0）
_C1 = (0.01 * 1.0) ** 2
_C2 = (0.03 * 1.0) ** 2


# ============================================
# 数据结构
# ============================================
@dataclass
class ImageInfo:
    """参与比较的图片元数据。"""

    path: str
    width: int
    height: int
    file_size: int

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


@dataclass
class SimilarPair:
    """一对相似图片及其 SSIM 相似度。"""

    first: ImageInfo
    second: ImageInfo
    score: float

    @property
    def difference(self) -> float:
        """相似度差值（``1 - SSIM``，对齐 AntiDuplX 的 compareThreshold 语义）。"""
        return 1.0 - self.score

    def involves(self, path: str) -> bool:
        return self.first.path == path or self.second.path == path


# ============================================
# 文件收集
# ============================================
def normalize_extensions(extensions: Optional[Iterable[str]]) -> Tuple[str, ...]:
    """把扩展名序列规范化为小写、带点、去重的元组。

    同时兼容单个逗号/分号分隔的字符串（如界面输入框的值）。
    """
    if not extensions:
        return IMAGE_EXTENSIONS

    if isinstance(extensions, str):
        raw_items: Iterable[str] = [extensions]
    else:
        raw_items = extensions

    normalized: List[str] = []
    for raw in raw_items:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        for piece in text.replace(";", ",").replace(" ", ",").split(","):
            item = piece.strip().lower()
            if not item:
                continue
            if not item.startswith("."):
                item = "." + item
            if item not in normalized:
                normalized.append(item)

    return tuple(normalized) or IMAGE_EXTENSIONS


def get_image_size(path: str) -> Optional[Tuple[int, int]]:
    """只读取图片宽高，不完整解码，失败时返回 None。"""
    if Image is None:
        raise RuntimeError("Pillow 未安装，无法读取图片尺寸")

    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None


def iter_image_files(
    directories: Sequence[str],
    recursive: bool = True,
    extensions: Optional[Iterable[str]] = None,
) -> List[str]:
    """遍历目录，返回符合扩展名的图片路径（去重并排序）。"""
    exts = normalize_extensions(extensions)
    found: List[str] = []
    seen = set()

    for directory in directories:
        if not directory or not os.path.isdir(directory):
            continue

        if recursive:
            walker: Iterable[Tuple[str, List[str], List[str]]] = os.walk(directory)
            for root, _dirs, files in walker:
                for name in files:
                    if name.lower().endswith(exts):
                        path = os.path.abspath(os.path.join(root, name))
                        if path not in seen:
                            seen.add(path)
                            found.append(path)
        else:
            for name in os.listdir(directory):
                full_path = os.path.abspath(os.path.join(directory, name))
                if os.path.isfile(full_path) and name.lower().endswith(exts):
                    if full_path not in seen:
                        seen.add(full_path)
                        found.append(full_path)

    found.sort()
    return found


def collect_image_infos(
    directories: Sequence[str],
    recursive: bool = True,
    extensions: Optional[Iterable[str]] = None,
    min_width: int = 0,
    min_height: int = 0,
    min_file_size: int = 0,
    message_cb: Optional[Callable[[str, str], None]] = None,
) -> List[ImageInfo]:
    """收集候选图片元数据，并按最小宽/高/文件大小过滤。"""
    paths = iter_image_files(directories, recursive=recursive, extensions=extensions)
    infos: List[ImageInfo] = []

    for path in paths:
        try:
            file_size = os.path.getsize(path)
        except OSError:
            continue

        if min_file_size and file_size < min_file_size:
            continue

        size = get_image_size(path)
        if size is None:
            if message_cb:
                message_cb(f"跳过无法解码的图片: {os.path.basename(path)}", "WARNING")
            continue

        width, height = size
        if min_width and width < min_width:
            continue
        if min_height and height < min_height:
            continue

        infos.append(ImageInfo(path=path, width=width, height=height, file_size=file_size))

    return infos


# ============================================
# 图像载入与 SSIM 计算
# ============================================
def _resampling_filter():
    resampling = getattr(Image, "Resampling", Image)
    return resampling.LANCZOS


def load_comparison_image(
    path: str,
    compare_size: int = DEFAULT_COMPARE_SIZE,
) -> Optional[np.ndarray]:
    """把图片解码为 ``compare_size x compare_size`` 的灰度 float32 数组（取值 0~1）。

    AntiDupl 的比较是尺度无关的：先把图片统一缩放到固定小尺寸，再比较内容，
    因此这里也做同样的归一化处理（不做保持宽高比的填充）。
    """
    if Image is None:
        raise RuntimeError("Pillow 未安装，无法加载图片")

    try:
        with Image.open(path) as img:
            gray = img.convert("L").resize((compare_size, compare_size), _resampling_filter())
            arr = np.asarray(gray, dtype=np.float32) / 255.0
    except Exception:
        return None

    if arr.ndim != 2 or arr.size == 0:
        return None

    return np.ascontiguousarray(arr, dtype=np.float32)


def gaussian_kernel_1d(size: int = SSIM_WINDOW_SIZE, sigma: float = SSIM_SIGMA) -> np.ndarray:
    """生成归一化的 1D 高斯核。"""
    size = max(3, int(size) | 1)  # 保证为不小于 3 的奇数
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-(coords ** 2) / (2.0 * float(sigma) ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def _window_size_for(shape: Sequence[int]) -> int:
    """根据图片尺寸选择合适的 SSIM 窗口（保证窗口不超过图像短边的一半）。"""
    smallest = min(int(shape[0]), int(shape[1]))
    size = SSIM_WINDOW_SIZE
    while size > smallest // 2 and size > 3:
        size -= 2
    return size


def _filter_1d(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """沿指定轴做可分离一维卷积（反射填充，支持任意前置批量维度）。"""
    length = kernel.size
    pad = length // 2
    moved = np.moveaxis(arr, axis, -1)
    pad_width = [(0, 0)] * (moved.ndim - 1) + [(pad, pad)]
    padded = np.pad(moved, pad_width, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, length, axis=-1)
    filtered = windows @ kernel
    return np.moveaxis(filtered, -1, axis)


def _filter_2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """可分离二维高斯滤波（支持任意前置批量维度）。"""
    return _filter_1d(_filter_1d(arr, kernel, -2), kernel, -1)


def prepare_ssim_features(
    gray: np.ndarray,
    kernel: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """预计算单张图片的局部均值与局部方差，供批量比较复用。"""
    if kernel is None:
        kernel = gaussian_kernel_1d(_window_size_for(gray.shape), SSIM_SIGMA)

    image = np.asarray(gray, dtype=np.float32)
    mu = _filter_2d(image, kernel)
    var = _filter_2d(image * image, kernel) - mu * mu
    return mu, np.maximum(var, 0.0)


def ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """计算两张同尺寸灰度图（取值 0~1）的 SSIM 相似度，范围约 -1~1。"""
    a = np.asarray(gray_a, dtype=np.float32)
    b = np.asarray(gray_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("SSIM 需要两张尺寸相同的图片")

    kernel = gaussian_kernel_1d(_window_size_for(a.shape), SSIM_SIGMA)
    mu_a, var_a = prepare_ssim_features(a, kernel)
    mu_b, var_b = prepare_ssim_features(b, kernel)
    cov = _filter_2d(a * b, kernel) - mu_a * mu_b

    numerator = (2.0 * mu_a * mu_b + _C1) * (2.0 * cov + _C2)
    denominator = (mu_a * mu_a + mu_b * mu_b + _C1) * (var_a + var_b + _C2)
    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def find_similar_pairs(
    images: Sequence[np.ndarray],
    infos: Sequence[ImageInfo],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> List[SimilarPair]:
    """在已归一化的图片集合中找出 SSIM 相似度不低于阈值的图片对。

    比较过程按「第一张图 × 其余图片」分批向量化，避免逐对 Python 循环。
    """
    count = len(images)
    if count < 2 or len(infos) != count:
        return []

    stacked = np.stack([np.asarray(img, dtype=np.float32) for img in images])
    height, width = stacked.shape[-2:]
    kernel = gaussian_kernel_1d(_window_size_for((height, width)), SSIM_SIGMA)

    means = _filter_2d(stacked, kernel)
    variances = np.maximum(_filter_2d(stacked * stacked, kernel) - means * means, 0.0)

    batch_size = max(1, _COMPARE_BATCH_PIXELS // (height * width))
    pairs: List[SimilarPair] = []

    for index in range(count - 1):
        if stop_event is not None and stop_event.is_set():
            break

        mu_a = means[index]
        var_a = variances[index]
        image_a = stacked[index]

        for start in range(index + 1, count, batch_size):
            stop = min(start + batch_size, count)
            if stop_event is not None and stop_event.is_set():
                break

            others = stacked[start:stop]
            cov = _filter_2d(image_a * others, kernel) - mu_a * means[start:stop]

            numerator = (2.0 * mu_a * means[start:stop] + _C1) * (2.0 * cov + _C2)
            denominator = (mu_a * mu_a + means[start:stop] ** 2 + _C1) * (
                var_a + variances[start:stop] + _C2
            )
            scores = (numerator / denominator).mean(axis=(-2, -1))

            hits = np.nonzero(scores >= threshold)[0]
            for offset in hits:
                pairs.append(
                    SimilarPair(
                        first=infos[index],
                        second=infos[start + int(offset)],
                        score=float(scores[offset]),
                    )
                )

        if progress_cb is not None:
            progress_cb(index + 1, count - 1)

    pairs.sort(key=lambda pair: pair.score, reverse=True)
    return pairs


def load_comparison_images(
    infos: Sequence[ImageInfo],
    compare_size: int = DEFAULT_COMPARE_SIZE,
    max_workers: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    stop_event: Optional[threading.Event] = None,
    message_cb: Optional[Callable[[str, str], None]] = None,
) -> Tuple[List[np.ndarray], List[ImageInfo]]:
    """多线程把图片载入为比较尺寸的灰度数组，返回（数组列表, 有效元数据列表）。"""
    images: Dict[int, np.ndarray] = {}
    infos_by_index: Dict[int, ImageInfo] = {}
    total = len(infos)
    if total == 0:
        return [], []

    workers = max_workers if max_workers and max_workers > 0 else (os.cpu_count() or 4)
    workers = min(workers, total)
    done = 0

    def load_one(index: int) -> Tuple[int, Optional[np.ndarray]]:
        return index, load_comparison_image(infos[index].path, compare_size)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_one, i): i for i in range(total)}
        for future in as_completed(futures):
            if stop_event is not None and stop_event.is_set():
                break

            try:
                index, array = future.result()
            except Exception:
                index, array = futures[future], None

            done += 1
            if array is not None:
                images[index] = array
                infos_by_index[index] = infos[index]
            elif message_cb is not None:
                message_cb(f"跳过无法解码的图片: {infos[index].name}", "WARNING")

            if progress_cb is not None:
                progress_cb(done, total)

    ordered = [i for i in range(total) if i in images]
    return [images[i] for i in ordered], [infos_by_index[i] for i in ordered]


def format_pair_line(pair: SimilarPair, delimiter: str = "\t") -> str:
    """按 AntiDuplX 输出风格格式化一行结果（相似度较低的一张在前）。"""
    first, second = pair.first, pair.second
    return delimiter.join(
        (
            first.path,
            second.path,
            f"{pair.score:.6f}",
            f"{pair.difference:.6f}",
        )
    )


def export_pairs(pairs: Sequence[SimilarPair], file_path: str) -> None:
    """把结果导出为文本文件（首行表头，便于人工核对）。"""
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(
            "first_image\tsecond_image\tssim_similarity\tdifference\n"
        )
        for pair in pairs:
            handle.write(format_pair_line(pair) + "\n")
