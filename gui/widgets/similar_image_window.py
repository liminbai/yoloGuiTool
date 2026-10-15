"""相似图像处理窗口（仿 AntiDupl，仅保留 SSIM 比较）。

界面组成参考 AntiDupl 主界面：

* 左侧：搜索目录列表、比较参数（SSIM 阈值 / 比较尺寸 / 扩展名 / 线程数 / 图片过滤）、
  操作按钮、日志与进度；
* 右侧：相似图片对结果表（双缩略图 + 相似度）与选中图片对的预览、删除操作。

比较算法只保留 SSIM，由 :mod:`gui.utils.image_similarity` 提供，
耗时搜索在 :class:`gui.threads.similar_image_thread.SimilarImageScanThread` 中执行。
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from PySide6.QtCore import QFile, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui.threads.similar_image_thread import SimilarImageScanThread
from gui.utils.image_similarity import (
    COMPARE_SIZE_CHOICES,
    DEFAULT_COMPARE_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    IMAGE_EXTENSIONS,
    SimilarPair,
    export_pairs,
    normalize_extensions,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover - 仅在未安装 Pillow 的环境触发
    Image = None


#: 结果表图标尺寸
THUMBNAIL_SIZE = 56
#: 预览图最大边长
PREVIEW_SIZE = 320


def _format_size(num_bytes: int) -> str:
    """把字节数格式化为易读文本。"""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _pil_to_pixmap(image) -> QPixmap:
    """把 Pillow 图片转换为 QPixmap。"""
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())


def load_thumbnail(path: str, max_size: int = THUMBNAIL_SIZE) -> QPixmap:
    """读取图片并生成等比缩放的缩略图，失败时返回空 QPixmap。"""
    if Image is None:
        return QPixmap()
    try:
        with Image.open(path) as img:
            thumb = img.copy()
        thumb.thumbnail((max_size, max_size))
        return _pil_to_pixmap(thumb)
    except Exception:
        return QPixmap()


class _SortableItem(QTableWidgetItem):
    """按自定义数值键排序的表格项（用于「序号」与「相似度」列）。"""

    def __init__(self, text: str = "", sort_key: Optional[float] = None):
        super().__init__(text)
        self._sort_key = sort_key

    def __lt__(self, other: QTableWidgetItem) -> bool:  # type: ignore[override]
        if isinstance(other, _SortableItem):
            left = self._sort_key if self._sort_key is not None else self.text()
            right = other._sort_key if other._sort_key is not None else other.text()
            try:
                return float(left) < float(right)
            except (TypeError, ValueError):
                pass
        return super().__lt__(other)


class SimilarImageWindow(QDialog):
    """相似图像处理窗口：配置 -> 搜索 -> 查看 -> 清理。"""

    #: 删除文件后发出（用于主窗口刷新状态栏）
    files_removed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("相似图像处理 - 基于 SSIM 的相似/重复图片查找（仿 AntiDupl）")
        self.setModal(False)
        self.resize(1240, 840)

        self._pairs: List[SimilarPair] = []
        self._thread: Optional[SimilarImageScanThread] = None
        self._thumbnails_loaded = set()

        self.init_ui()

    # ============================================
    # 界面构建
    # ============================================
    def init_ui(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(8)

        hint = QLabel(
            "将图片统一缩放到相同比较尺寸后计算 SSIM 结构相似度，"
            "相似度不低于阈值的图片对视为相似/重复图片。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; padding: 2px;")
        outer_layout.addWidget(hint)

        if Image is None:
            warning = QLabel("⚠️ 未检测到 Pillow，无法载入图片，请先安装：pip install pillow")
            warning.setStyleSheet("color: #c0392b; font-weight: bold;")
            outer_layout.addWidget(warning)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._create_left_panel())
        splitter.addWidget(self._create_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 880])
        outer_layout.addWidget(splitter, stretch=1)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._create_folder_group())
        layout.addWidget(self._create_options_group())
        layout.addWidget(self._create_action_group())
        layout.addWidget(self._create_log_group(), stretch=1)
        return panel

    def _create_folder_group(self) -> QGroupBox:
        group = QGroupBox("📁 搜索目录")
        group.setStyleSheet("font-size: 13px; color: #2c3e50;")
        layout = QVBoxLayout(group)

        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.folder_list.setMinimumHeight(96)
        layout.addWidget(self.folder_list)

        button_layout = QHBoxLayout()
        self.add_folder_btn = QPushButton("➕ 添加目录")
        self.add_folder_btn.clicked.connect(self.add_folder)
        button_layout.addWidget(self.add_folder_btn)

        self.remove_folder_btn = QPushButton("➖ 移除选中")
        self.remove_folder_btn.clicked.connect(self.remove_selected_folders)
        button_layout.addWidget(self.remove_folder_btn)

        self.clear_folder_btn = QPushButton("🧹 清空")
        self.clear_folder_btn.clicked.connect(self.folder_list.clear)
        button_layout.addWidget(self.clear_folder_btn)

        layout.addLayout(button_layout)

        self.recursive_check = QCheckBox("包含子目录（AntiDuplX -sd=1）")
        self.recursive_check.setChecked(True)
        layout.addWidget(self.recursive_check)

        return group

    def _create_options_group(self) -> QGroupBox:
        group = QGroupBox("⚙️ 比较参数")
        group.setStyleSheet("font-size: 13px; color: #2c3e50;")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.50, 1.00)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(DEFAULT_SIMILARITY_THRESHOLD)
        self.threshold_spin.setToolTip(
            "SSIM 相似度阈值，越高越严格。0.95 等价于 AntiDuplX 的 -ct=0.05（相似度差值）。"
        )
        form.addRow("SSIM 相似度阈值:", self.threshold_spin)

        self.compare_size_combo = QComboBox()
        for size in COMPARE_SIZE_CHOICES:
            self.compare_size_combo.addItem(f"{size} x {size}", size)
        default_index = (
            COMPARE_SIZE_CHOICES.index(DEFAULT_COMPARE_SIZE)
            if DEFAULT_COMPARE_SIZE in COMPARE_SIZE_CHOICES
            else 0
        )
        self.compare_size_combo.setCurrentIndex(default_index)
        self.compare_size_combo.setToolTip(
            "计算 SSIM 前把图片缩放到的比较尺寸（AntiDuplX -cs，默认 32x32）。"
        )
        form.addRow("比较尺寸:", self.compare_size_combo)

        self.extensions_edit = QLineEdit(",".join(IMAGE_EXTENSIONS))
        self.extensions_edit.setToolTip("参与搜索的图片扩展名，逗号分隔（AntiDuplX -ie）。")
        form.addRow("图片扩展名:", self.extensions_edit)

        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 64)
        self.threads_spin.setValue(0)
        self.threads_spin.setSpecialValueText("自动")
        self.threads_spin.setToolTip("载入图片的线程数，0 表示自动使用 CPU 核心数。")
        form.addRow("载入线程数:", self.threads_spin)

        self.min_width_spin = QSpinBox()
        self.min_width_spin.setRange(0, 100000)
        self.min_width_spin.setValue(0)
        self.min_width_spin.setSpecialValueText("不限")
        form.addRow("最小宽度:", self.min_width_spin)

        self.min_height_spin = QSpinBox()
        self.min_height_spin.setRange(0, 100000)
        self.min_height_spin.setValue(0)
        self.min_height_spin.setSpecialValueText("不限")
        form.addRow("最小高度:", self.min_height_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(0, 1024 * 1024)
        self.min_size_spin.setValue(0)
        self.min_size_spin.setSuffix(" KB")
        self.min_size_spin.setSpecialValueText("不限")
        form.addRow("最小文件大小:", self.min_size_spin)

        return group

    def _create_action_group(self) -> QGroupBox:
        group = QGroupBox("▶ 操作")
        group.setStyleSheet("font-size: 13px; color: #2c3e50;")
        layout = QVBoxLayout(group)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("🔍 开始搜索")
        self.start_btn.setMinimumHeight(34)
        self.start_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2980b9; color: white; font-weight: bold;
                padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #2471a3; }
            QPushButton:disabled { background-color: #95a5a6; }
            """
        )
        self.start_btn.clicked.connect(self.start_search)
        button_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹ 停止")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_search)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status_label = QLabel("就绪")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        return group

    def _create_log_group(self) -> QGroupBox:
        group = QGroupBox("📝 日志")
        group.setStyleSheet("font-size: 13px; color: #2c3e50;")
        layout = QVBoxLayout(group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(110)
        layout.addWidget(self.log_text)

        return group

    def _create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._create_result_group(), stretch=1)
        layout.addWidget(self._create_preview_group())
        return panel

    def _create_result_group(self) -> QGroupBox:
        group = QGroupBox("🏆 相似图片对")
        group.setStyleSheet("font-size: 13px; color: #2c3e50;")
        layout = QVBoxLayout(group)

        self.result_table = QTableWidget(0, 6)
        self.result_table.setHorizontalHeaderLabels(
            ["#", "图片 A", "文件 A", "图片 B", "文件 B", "相似度"]
        )
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setIconSize(QSize(THUMBNAIL_SIZE, THUMBNAIL_SIZE))
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.result_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.result_table.setSortingEnabled(True)
        self.result_table.itemSelectionChanged.connect(self.on_selection_changed)
        self.result_table.verticalScrollBar().valueChanged.connect(
            self.update_visible_thumbnails
        )
        self.result_table.horizontalHeader().sortIndicatorChanged.connect(
            self.renumber_rows
        )

        header = self.result_table.horizontalHeader()
        for column, mode in (
            (0, QHeaderView.ResizeToContents),
            (1, QHeaderView.ResizeToContents),
            (2, QHeaderView.Stretch),
            (3, QHeaderView.ResizeToContents),
            (4, QHeaderView.Stretch),
            (5, QHeaderView.ResizeToContents),
        ):
            header.setSectionResizeMode(column, mode)

        layout.addWidget(self.result_table)

        bottom_layout = QHBoxLayout()
        self.count_label = QLabel("共 0 组")
        bottom_layout.addWidget(self.count_label)
        bottom_layout.addStretch()

        self.export_btn = QPushButton("📤 导出结果")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        bottom_layout.addWidget(self.export_btn)

        self.clear_result_btn = QPushButton("🧹 清空结果")
        self.clear_result_btn.clicked.connect(self.clear_results)
        bottom_layout.addWidget(self.clear_result_btn)

        layout.addLayout(bottom_layout)
        return group

    def _create_preview_group(self) -> QGroupBox:
        group = QGroupBox("🖼 选中图片对预览")
        group.setStyleSheet("font-size: 13px; color: #2c3e50;")
        layout = QHBoxLayout(group)

        self.preview_left = QLabel("请选择左侧结果表中的一行")
        self.preview_right = QLabel("")
        for label in (self.preview_left, self.preview_right):
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumSize(PREVIEW_SIZE // 2, PREVIEW_SIZE // 2)
            label.setStyleSheet("border: 1px solid #dfe6e9; background: #fafafa;")

        self.preview_left_info = QLabel("")
        self.preview_right_info = QLabel("")
        for label in (self.preview_left_info, self.preview_right_info):
            label.setWordWrap(True)
            label.setStyleSheet("color: #34495e; font-size: 12px;")

        left_column = QVBoxLayout()
        left_column.addWidget(self.preview_left)
        left_column.addWidget(self.preview_left_info)

        right_column = QVBoxLayout()
        right_column.addWidget(self.preview_right)
        right_column.addWidget(self.preview_right_info)

        action_column = QVBoxLayout()
        self.trash_check = QCheckBox("删除到回收站")
        self.trash_check.setChecked(True)
        self.trash_check.setToolTip("取消勾选将永久删除文件（不可恢复）。")
        action_column.addWidget(self.trash_check)

        self.delete_left_btn = QPushButton("🗑 删除图片 A")
        self.delete_left_btn.clicked.connect(lambda: self.delete_selected("first"))
        action_column.addWidget(self.delete_left_btn)

        self.delete_right_btn = QPushButton("🗑 删除图片 B")
        self.delete_right_btn.clicked.connect(lambda: self.delete_selected("second"))
        action_column.addWidget(self.delete_right_btn)

        self.delete_both_btn = QPushButton("🗑 删除两张")
        self.delete_both_btn.clicked.connect(lambda: self.delete_selected("both"))
        action_column.addWidget(self.delete_both_btn)

        action_column.addStretch()
        for button in (self.delete_left_btn, self.delete_right_btn, self.delete_both_btn):
            button.setEnabled(False)

        layout.addLayout(left_column, stretch=1)
        layout.addLayout(right_column, stretch=1)
        layout.addLayout(action_column)
        return group

    # ============================================
    # 目录管理
    # ============================================
    def add_folder(self) -> None:
        """添加搜索目录。"""
        directory = QFileDialog.getExistingDirectory(self, "选择搜索目录", "")
        if not directory:
            return
        self.add_folder_path(directory)

    def add_folder_path(self, directory: str) -> None:
        """把目录加入列表（自动去重）。"""
        existing = {
            self.folder_list.item(i).text() for i in range(self.folder_list.count())
        }
        normalized = os.path.abspath(directory)
        if normalized in existing:
            self.append_log(f"目录已在列表中: {normalized}", "WARNING")
            return
        self.folder_list.addItem(normalized)

    def remove_selected_folders(self) -> None:
        """移除列表中选中的目录。"""
        for item in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(item))

    def folders(self) -> List[str]:
        return [self.folder_list.item(i).text() for i in range(self.folder_list.count())]

    # ============================================
    # 搜索流程
    # ============================================
    def set_options(
        self,
        threshold: Optional[float] = None,
        compare_size: Optional[int] = None,
        recursive: Optional[bool] = None,
        extensions: Optional[Sequence[str]] = None,
    ) -> None:
        """供外部（如主窗口）预设比较参数。"""
        if threshold is not None:
            self.threshold_spin.setValue(float(threshold))
        if compare_size is not None and compare_size in COMPARE_SIZE_CHOICES:
            self.compare_size_combo.setCurrentIndex(
                COMPARE_SIZE_CHOICES.index(int(compare_size))
            )
        if recursive is not None:
            self.recursive_check.setChecked(bool(recursive))
        if extensions:
            self.extensions_edit.setText(",".join(normalize_extensions(extensions)))

    def start_search(self) -> None:
        """按当前参数启动后台搜索。"""
        directories = self.folders()
        if not directories:
            QMessageBox.warning(self, "提示", "请先添加至少一个搜索目录。")
            return

        if Image is None:
            QMessageBox.critical(self, "缺少依赖", "未检测到 Pillow，请先安装：pip install pillow")
            return

        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "提示", "搜索正在进行中。")
            return

        self.clear_results()
        self.log_text.clear()
        self.progress.setRange(0, 0)
        self.status_label.setText("正在搜索...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self._thread = SimilarImageScanThread(
            directories=directories,
            compare_size=self.compare_size_combo.currentData(),
            threshold=self.threshold_spin.value(),
            extensions=normalize_extensions(self.extensions_edit.text()),
            recursive=self.recursive_check.isChecked(),
            max_workers=self.threads_spin.value(),
            min_width=self.min_width_spin.value(),
            min_height=self.min_height_spin.value(),
            min_file_size=self.min_size_spin.value() * 1024,
            parent=self,
        )
        self._thread.loading_progress.connect(self.on_loading_progress)
        self._thread.comparing_progress.connect(self.on_comparing_progress)
        self._thread.message.connect(self.append_log)
        self._thread.results.connect(self.on_results)
        self._thread.failed.connect(self.on_failed)
        self._thread.finished.connect(self.on_thread_finished)
        self._thread.start()

    def stop_search(self) -> None:
        """请求停止搜索。"""
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop()
            self.status_label.setText("正在停止...")
            self.stop_btn.setEnabled(False)

    def on_loading_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        self.status_label.setText(f"正在载入图片 {current}/{total}...")

    def on_comparing_progress(self, current: int, total: int) -> None:
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(current)
        self.status_label.setText(f"正在比较图片 {current}/{total}...")

    def on_results(self, pairs: Sequence[SimilarPair]) -> None:
        """接收搜索结果并填充表格。"""
        self._pairs = list(pairs)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.populate_table()
        self.count_label.setText(f"共 {len(self._pairs)} 组")
        self.export_btn.setEnabled(bool(self._pairs))
        if not self._pairs:
            self.status_label.setText("未找到相似图片，可尝试降低相似度阈值。")
        else:
            self.status_label.setText(f"搜索完成，共找到 {len(self._pairs)} 组相似图片。")

    def on_failed(self, message: str) -> None:
        self.append_log(f"搜索失败: {message}", "ERROR")
        self.status_label.setText("搜索失败")
        QMessageBox.critical(self, "错误", f"搜索失败:\n{message}")

    def on_thread_finished(self) -> None:
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
        self._thread = None

    # ============================================
    # 结果表
    # ============================================
    def populate_table(self) -> None:
        """根据 self._pairs 重建结果表。"""
        table = self.result_table
        table.setSortingEnabled(False)
        table.clearContents()
        table.setRowCount(len(self._pairs))
        self._thumbnails_loaded.clear()

        for row, pair in enumerate(self._pairs):
            index_item = _SortableItem(str(row + 1), float(row + 1))
            # UserRole 固定保存该行对应的 self._pairs 下标，排序后仍可正确定位
            index_item.setData(Qt.UserRole, row)
            table.setItem(row, 0, index_item)

            first_icon = QTableWidgetItem()
            first_icon.setData(Qt.UserRole, pair.first.path)
            table.setItem(row, 1, first_icon)

            table.setItem(row, 2, self._make_path_item(pair.first))

            second_icon = QTableWidgetItem()
            second_icon.setData(Qt.UserRole, pair.second.path)
            table.setItem(row, 3, second_icon)

            table.setItem(row, 4, self._make_path_item(pair.second))

            score_item = _SortableItem(f"{pair.score * 100:.2f}%", pair.score)
            score_item.setForeground(Qt.darkGreen)
            table.setItem(row, 5, score_item)

        table.setSortingEnabled(True)
        table.sortItems(5, Qt.DescendingOrder)
        self.renumber_rows()
        self.update_visible_thumbnails()

    @staticmethod
    def _make_path_item(info) -> QTableWidgetItem:
        item = QTableWidgetItem(info.name)
        item.setToolTip(
            f"{info.path}\n尺寸: {info.width} x {info.height}\n大小: {_format_size(info.file_size)}"
        )
        return item

    def renumber_rows(self, *args) -> None:
        """按当前显示顺序重排序号列（重排序号期间临时关闭排序，避免反复重排）。"""
        table = self.result_table
        sorting_enabled = table.isSortingEnabled()
        table.setSortingEnabled(False)
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if isinstance(item, _SortableItem):
                item.setText(str(row + 1))
                item._sort_key = float(row + 1)
        table.setSortingEnabled(sorting_enabled)

    def update_visible_thumbnails(self) -> None:
        """只为可见行加载缩略图，避免大结果集卡顿。"""
        table = self.result_table
        if table.rowCount() == 0:
            return

        viewport = table.viewport()
        top = table.rowAt(0)
        bottom = table.rowAt(viewport.height() - 1)
        if top < 0:
            top = 0
        if bottom < 0:
            bottom = table.rowCount() - 1

        for row in range(max(0, top), min(bottom, table.rowCount() - 1) + 1):
            for column in (1, 3):
                item = table.item(row, column)
                if item is None:
                    continue
                path = item.data(Qt.UserRole)
                if not path or path in self._thumbnails_loaded:
                    continue
                self._thumbnails_loaded.add(path)
                pixmap = load_thumbnail(path, THUMBNAIL_SIZE)
                if not pixmap.isNull():
                    item.setIcon(QIcon(pixmap))

    def selected_pair(self) -> Optional[SimilarPair]:
        """返回当前选中的相似图片对。"""
        row = self.result_table.currentRow()
        if row < 0:
            return None

        index_item = self.result_table.item(row, 0)
        if index_item is None:
            return None

        pair_index = index_item.data(Qt.UserRole)
        if pair_index is None or not 0 <= int(pair_index) < len(self._pairs):
            return None

        return self._pairs[int(pair_index)]

    def on_selection_changed(self) -> None:
        pair = self.selected_pair()
        has_selection = pair is not None
        for button in (self.delete_left_btn, self.delete_right_btn, self.delete_both_btn):
            button.setEnabled(has_selection)

        if pair is None:
            self.preview_left.setText("请选择左侧结果表中的一行")
            self.preview_right.clear()
            self.preview_left_info.clear()
            self.preview_right_info.clear()
            return

        self._show_preview(self.preview_left, self.preview_left_info, pair.first)
        self._show_preview(self.preview_right, self.preview_right_info, pair.second)

    def _show_preview(self, label: QLabel, info_label: QLabel, info) -> None:
        if not os.path.exists(info.path):
            label.setPixmap(QPixmap())
            label.setText("文件不存在")
            info_label.setText("")
            return

        pixmap = load_thumbnail(info.path, PREVIEW_SIZE)
        if pixmap.isNull():
            label.setPixmap(QPixmap())
            label.setText("无法预览")
        else:
            label.setText("")
            label.setPixmap(pixmap)

        info_label.setText(
            f"{info.name}\n{info.width} x {info.height} · {_format_size(info.file_size)}\n{info.path}"
        )

    # ============================================
    # 删除与导出
    # ============================================
    def delete_selected(self, side: str) -> None:
        """删除选中图片对中的 A/B/两张图片。"""
        pair = self.selected_pair()
        if pair is None:
            return

        if side == "first":
            targets = [pair.first.path]
        elif side == "second":
            targets = [pair.second.path]
        else:
            targets = [pair.first.path, pair.second.path]

        names = "\n".join(os.path.basename(path) for path in targets)
        action = "移入回收站" if self.trash_check.isChecked() else "永久删除"
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定要{action}以下 {len(targets)} 个文件吗？\n\n{names}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        removed: List[str] = []
        failures: List[str] = []
        for path in targets:
            ok, detail = self._remove_file(path)
            if ok:
                removed.append(path)
                self.append_log(f"{detail}: {path}", "SUCCESS")
            else:
                failures.append(f"{os.path.basename(path)}: {detail}")

        if failures:
            QMessageBox.warning(self, "部分文件未删除", "\n".join(failures))

        if removed:
            self._pairs = [p for p in self._pairs if not any(p.involves(r) for r in removed)]
            self.populate_table()
            self.count_label.setText(f"共 {len(self._pairs)} 组")
            self.export_btn.setEnabled(bool(self._pairs))
            self.files_removed.emit(len(removed))

    def _remove_file(self, path: str):
        """删除文件：优先使用回收站，返回 (是否成功, 说明)。"""
        if not os.path.exists(path):
            return True, "文件已不存在"

        if self.trash_check.isChecked():
            try:
                if QFile.moveToTrash(path):
                    return True, "已移入回收站"
            except Exception:
                pass
            return False, "回收站不可用，可取消勾选「删除到回收站」后永久删除"

        try:
            os.remove(path)
            return True, "已永久删除"
        except OSError as exc:
            return False, str(exc)

    def export_results(self) -> None:
        """把结果导出为文本文件（对齐 AntiDuplX -of 输出）。"""
        if not self._pairs:
            QMessageBox.information(self, "提示", "当前没有可导出的结果。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出相似图片结果", "similar_images.txt", "文本文件 (*.txt);;CSV文件 (*.csv)"
        )
        if not file_path:
            return

        try:
            export_pairs(self._pairs, file_path)
            self.append_log(f"结果已导出到: {file_path}", "SUCCESS")
            QMessageBox.information(self, "成功", f"结果已导出到:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "错误", f"导出失败:\n{exc}")

    def clear_results(self) -> None:
        """清空结果表与缓存。"""
        self._pairs = []
        self._thumbnails_loaded.clear()
        self.result_table.clearContents()
        self.result_table.setRowCount(0)
        self.count_label.setText("共 0 组")
        self.export_btn.setEnabled(False)
        self.on_selection_changed()

    # ============================================
    # 日志与关闭
    # ============================================
    def append_log(self, message: str, level: str = "INFO") -> None:
        """追加一条日志（带级别前缀与颜色提示）。"""
        prefix = {"SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "ℹ️")
        self.log_text.append(f"{prefix} {message}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        """关闭时若仍在搜索则先停止线程。"""
        if self._thread is not None and self._thread.isRunning():
            self._thread.stop()
            self._thread.wait(5000)
        super().closeEvent(event)
