"""相似图像搜索线程（基于 SSIM，仿 AntiDupl）。

线程负责：收集候选图片 -> 多线程载入并归一化 -> 逐对计算 SSIM -> 回传结果。
所有耗时操作都在工作线程内完成，界面只通过信号更新，保证主界面不卡顿。
"""

from __future__ import annotations

import threading
from typing import Optional, Sequence

try:
    from PySide6.QtCore import QThread, Signal
except ImportError:  # pragma: no cover - 兼容没有 Qt 的测试环境
    from threading import Thread

    class QThread(Thread):
        def __init__(self, parent=None):
            super().__init__()

        def isRunning(self):  # noqa: N802 - 与 Qt 接口保持一致
            return self.is_alive()

        def wait(self, *args, **kwargs):
            self.join()

    class Signal:
        def __init__(self, *args, **kwargs):
            self._slots = []

        def connect(self, slot):
            self._slots.append(slot)

        def emit(self, *args, **kwargs):
            for slot in self._slots:
                slot(*args, **kwargs)


from gui.utils.image_similarity import (
    DEFAULT_COMPARE_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    collect_image_infos,
    find_similar_pairs,
    load_comparison_images,
)


class SimilarImageScanThread(QThread):
    """在后台执行 SSIM 相似图片搜索。"""

    loading_progress = Signal(int, int)  # 已载入, 总数
    comparing_progress = Signal(int, int)  # 已比较, 总组数
    message = Signal(str, str)  # 消息, 级别
    results = Signal(object)  # List[SimilarPair]
    failed = Signal(str)  # 错误信息

    def __init__(
        self,
        directories: Sequence[str],
        compare_size: int = DEFAULT_COMPARE_SIZE,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
        max_workers: int = 0,
        min_width: int = 0,
        min_height: int = 0,
        min_file_size: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.directories = [str(d) for d in directories if d]
        self.compare_size = int(compare_size) if compare_size else DEFAULT_COMPARE_SIZE
        self.threshold = float(threshold)
        self.extensions = extensions
        self.recursive = bool(recursive)
        self.max_workers = int(max_workers or 0)
        self.min_width = int(min_width or 0)
        self.min_height = int(min_height or 0)
        self.min_file_size = int(min_file_size or 0)
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """请求停止搜索（当前批次结束后退出）。"""
        self._stop_event.set()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def run(self) -> None:  # noqa: D401 - QThread 入口
        try:
            if not self.directories:
                self.message.emit("请先添加至少一个搜索目录", "WARNING")
                self.results.emit([])
                return

            self.message.emit("正在扫描目录，收集候选图片...", "INFO")
            infos = collect_image_infos(
                self.directories,
                recursive=self.recursive,
                extensions=self.extensions,
                min_width=self.min_width,
                min_height=self.min_height,
                min_file_size=self.min_file_size,
                message_cb=self.message.emit,
            )

            if self._stop_event.is_set():
                self.message.emit("搜索已停止", "WARNING")
                self.results.emit([])
                return

            self.message.emit(f"共收集到 {len(infos)} 张候选图片", "INFO")
            if len(infos) < 2:
                self.message.emit("候选图片少于 2 张，无法进行相似度比较", "WARNING")
                self.results.emit([])
                return

            total_pairs = len(infos) * (len(infos) - 1) // 2
            self.message.emit(f"待比较图片对约 {total_pairs:,} 组，正在载入图片...", "INFO")

            images, valid_infos = load_comparison_images(
                infos,
                compare_size=self.compare_size,
                max_workers=self.max_workers,
                progress_cb=self.loading_progress.emit,
                stop_event=self._stop_event,
                message_cb=self.message.emit,
            )

            if self._stop_event.is_set():
                self.message.emit("搜索已停止", "WARNING")
                self.results.emit([])
                return

            if len(images) < 2:
                self.message.emit("有效图片少于 2 张，无法进行相似度比较", "WARNING")
                self.results.emit([])
                return

            self.message.emit(
                f"开始 SSIM 比较：{len(images)} 张图片，相似度阈值 {self.threshold:.2f}",
                "INFO",
            )
            pairs = find_similar_pairs(
                images,
                valid_infos,
                threshold=self.threshold,
                progress_cb=self.comparing_progress.emit,
                stop_event=self._stop_event,
            )

            if self._stop_event.is_set():
                self.message.emit(
                    f"搜索已停止，已找到 {len(pairs)} 组相似图片", "WARNING"
                )
            else:
                self.message.emit(f"搜索完成，共找到 {len(pairs)} 组相似图片", "SUCCESS")

            self.results.emit(list(pairs))
        except Exception as exc:  # pragma: no cover - 防御性处理
            self.failed.emit(str(exc))
