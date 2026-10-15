"""相似图像处理（SSIM）工具测试。

覆盖 ``gui/utils/image_similarity.py`` 的比较算法与 ``gui/threads/similar_image_thread.py``
的搜索流程，全部测试只依赖 numpy / Pillow，不需要 Qt 事件循环。
"""

import os
import sys
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from PIL import Image

from gui.threads.similar_image_thread import SimilarImageScanThread
from gui.utils.image_similarity import (
    DEFAULT_COMPARE_SIZE,
    IMAGE_EXTENSIONS,
    ImageInfo,
    SimilarPair,
    collect_image_infos,
    export_pairs,
    find_similar_pairs,
    load_comparison_image,
    load_comparison_images,
    normalize_extensions,
    ssim,
)


def _save_image(path, array):
    Image.fromarray(array).save(path)
    return str(path)


def _gradient(width=160, height=120):
    """生成一张有明确结构的梯度图，便于构造相似图片。"""
    x = np.linspace(0, 255, width, dtype=np.float32)
    y = np.linspace(0, 255, height, dtype=np.float32)
    blue = np.tile(x, (height, 1))
    green = np.tile(y[:, None], (1, width))
    return np.stack([blue, green, np.full_like(blue, 90.0)], axis=-1).astype(np.uint8)


# ============================================
# 扩展名与文件收集
# ============================================
def test_normalize_extensions_accepts_string_and_sequence():
    assert normalize_extensions(".JPG; png, .webp") == (".jpg", ".png", ".webp")
    assert normalize_extensions([".jpg", "PNG", ".jpg"]) == (".jpg", ".png")
    assert normalize_extensions("") == IMAGE_EXTENSIONS
    assert normalize_extensions(None) == IMAGE_EXTENSIONS


def test_iter_image_files_respects_recursion_and_extensions(tmp_path):
    from gui.utils.image_similarity import iter_image_files

    nested = tmp_path / "nested"
    nested.mkdir()
    _save_image(tmp_path / "top.png", _gradient())
    _save_image(nested / "deep.JPG", _gradient())
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

    recursive = iter_image_files([str(tmp_path)], recursive=True)
    assert len(recursive) == 2
    assert all(p.lower().endswith((".png", ".jpg")) for p in recursive)

    flat = iter_image_files([str(tmp_path)], recursive=False)
    assert len(flat) == 1
    assert flat[0].endswith("top.png")

    # 目录重叠时不应重复收集
    deduped = iter_image_files([str(tmp_path), str(nested)], recursive=True)
    assert len(deduped) == 2

    # 扩展名过滤
    png_only = iter_image_files([str(tmp_path)], recursive=True, extensions=[".png"])
    assert len(png_only) == 1


def test_collect_image_infos_applies_filters_and_reports_broken_files(tmp_path):
    _save_image(tmp_path / "big.png", _gradient(300, 200))
    _save_image(tmp_path / "small.png", _gradient(40, 30))
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not-an-image")

    messages = []
    infos = collect_image_infos(
        [str(tmp_path)],
        recursive=True,
        min_width=100,
        message_cb=lambda message, level: messages.append((level, message)),
    )

    assert [info.name for info in infos] == ["big.png"]
    assert infos[0].width == 300 and infos[0].height == 200
    assert any(level == "WARNING" and "broken.png" in message for level, message in messages)

    # 最小文件大小过滤掉小图
    assert collect_image_infos([str(tmp_path)], min_file_size=10**7) == []


# ============================================
# 图像载入与 SSIM
# ============================================
def test_load_comparison_image_normalizes_size_and_range(tmp_path):
    path = _save_image(tmp_path / "img.png", _gradient(240, 120))
    array = load_comparison_image(path, compare_size=32)

    assert array.shape == (32, 32)
    assert array.dtype == np.float32
    assert 0.0 <= float(array.min()) and float(array.max()) <= 1.0


def test_load_comparison_image_returns_none_for_undecodable_file(tmp_path):
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"still-not-an-image")
    assert load_comparison_image(str(broken), DEFAULT_COMPARE_SIZE) is None


def test_ssim_identical_and_unrelated_images():
    gradient = np.asarray(Image.fromarray(_gradient(64, 64)).convert("L"), dtype=np.float32) / 255.0
    assert ssim(gradient, gradient) == pytest.approx(1.0)

    other = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    assert ssim(gradient, other) < 0.9

    # 纯黑与纯白的亮度差异应导致接近 0 的相似度
    black = np.zeros((32, 32), dtype=np.float32)
    white = np.ones((32, 32), dtype=np.float32)
    assert ssim(black, white) < 0.01


def test_ssim_requires_matching_shapes():
    with pytest.raises(ValueError):
        ssim(np.zeros((16, 16), dtype=np.float32), np.zeros((16, 8), dtype=np.float32))


# ============================================
# 相似对搜索
# ============================================
def _build_dataset(tmp_path):
    """构造一组图片：3 张相似（缩放/JPEG/轻微改动）+ 3 张无关。"""
    gradient = _gradient(200, 150)
    paths = [
        _save_image(tmp_path / "base.png", gradient),
        _save_image(tmp_path / "base_scaled.png", _gradient(100, 75)),
    ]
    Image.fromarray(gradient).save(tmp_path / "base.jpg", quality=90)
    paths.append(str(tmp_path / "base.jpg"))

    rng = np.random.default_rng(5)
    for index in range(3):
        paths.append(
            _save_image(
                tmp_path / f"noise{index}.png",
                (rng.random((150, 200, 3)) * 255).astype(np.uint8),
            )
        )
    return paths


def test_find_similar_pairs_finds_scaled_and_jpeg_duplicates(tmp_path):
    _build_dataset(tmp_path)
    infos = collect_image_infos([str(tmp_path)])
    images, valid_infos = load_comparison_images(infos, DEFAULT_COMPARE_SIZE, max_workers=2)

    pairs = find_similar_pairs(images, valid_infos, threshold=0.90)

    names = {frozenset((pair.first.name, pair.second.name)) for pair in pairs}
    assert frozenset(("base.png", "base_scaled.png")) in names
    assert frozenset(("base.png", "base.jpg")) in names
    assert frozenset(("base_scaled.png", "base.jpg")) in names
    assert all("noise" not in name for pair in names for name in pair)

    scores = [pair.score for pair in pairs]
    assert scores == sorted(scores, reverse=True)
    assert all(pair.difference == pytest.approx(1.0 - pair.score) for pair in pairs)


def test_find_similar_pairs_respects_threshold_and_stop_event(tmp_path):
    _build_dataset(tmp_path)
    infos = collect_image_infos([str(tmp_path)])
    images, valid_infos = load_comparison_images(infos, DEFAULT_COMPARE_SIZE, max_workers=2)

    assert find_similar_pairs(images, valid_infos, threshold=0.9999999) == []

    stop_event = threading.Event()
    stop_event.set()
    assert find_similar_pairs(images, valid_infos, threshold=0.0, stop_event=stop_event) == []

    # 进度回调应报告累计处理行数
    progress = []
    find_similar_pairs(
        images,
        valid_infos,
        threshold=0.99,
        progress_cb=lambda current, total: progress.append((current, total)),
    )
    assert progress
    assert progress[-1][0] == len(images) - 1


def test_load_comparison_images_skips_broken_and_reports(tmp_path):
    _save_image(tmp_path / "ok.png", _gradient())
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"broken")

    infos = [
        ImageInfo(path=str(tmp_path / "ok.png"), width=160, height=120, file_size=1),
        ImageInfo(path=str(broken), width=0, height=0, file_size=1),
    ]
    messages = []
    images, valid = load_comparison_images(
        infos, DEFAULT_COMPARE_SIZE, max_workers=2,
        message_cb=lambda m, level: messages.append(level),
    )

    assert len(images) == 1 and len(valid) == 1
    assert valid[0].name == "ok.png"
    assert messages == ["WARNING"]


# ============================================
# 导出
# ============================================
def test_export_pairs_writes_header_and_rows(tmp_path):
    pair = SimilarPair(
        first=ImageInfo("/a/one.png", 10, 10, 100),
        second=ImageInfo("/b/two.jpg", 20, 20, 200),
        score=0.9876543,
    )
    out_file = tmp_path / "result.txt"
    export_pairs([pair], str(out_file))

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("first_image")
    assert lines[1].split("\t") == ["/a/one.png", "/b/two.jpg", "0.987654", "0.012346"]


# ============================================
# 搜索线程（同步执行 run()，避免依赖 Qt 事件循环）
# ============================================
def test_scan_thread_reports_results(tmp_path):
    _build_dataset(tmp_path)
    thread = SimilarImageScanThread(
        directories=[str(tmp_path)], compare_size=DEFAULT_COMPARE_SIZE, threshold=0.90
    )
    results = []
    messages = []
    thread.results.connect(results.append)
    thread.message.connect(lambda message, level: messages.append((level, message)))

    thread.run()

    assert results and results[0]
    assert all(isinstance(pair, SimilarPair) for pair in results[0])
    assert messages[-1][0] == "SUCCESS"
    assert any("SSIM" in message for _level, message in messages)


def test_scan_thread_requires_directories_and_two_images(tmp_path):
    empty_thread = SimilarImageScanThread(directories=[], threshold=0.9)
    empty_results = []
    empty_thread.results.connect(empty_results.append)
    empty_thread.run()
    assert empty_results == [[]]

    _save_image(tmp_path / "only.png", _gradient())
    single_thread = SimilarImageScanThread(directories=[str(tmp_path)], threshold=0.9)
    single_results = []
    single_thread.results.connect(single_results.append)
    single_thread.run()
    assert single_results == [[]]


def test_scan_thread_stop_before_run_returns_no_results(tmp_path):
    _build_dataset(tmp_path)
    thread = SimilarImageScanThread(directories=[str(tmp_path)], threshold=0.90)
    thread.stop()
    assert thread.is_stopped()

    results = []
    messages = []
    thread.results.connect(results.append)
    thread.message.connect(lambda message, level: messages.append((level, message)))
    thread.run()

    assert results == [[]]
    assert any("停止" in message for _level, message in messages)
