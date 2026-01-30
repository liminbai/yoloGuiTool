import sys
from PySide6.QtWidgets import QWidget, QHBoxLayout, QComboBox

class ComboBox(QWidget):
    def __init__(self, parent=None, items=None):
        # 在 Python 3 中，super() 不需要显式传递类名和 self
        super().__init__(parent)

        layout = QHBoxLayout()
        self.cb = QComboBox()
        
        # 处理默认参数为列表的情况（避免使用可变对象作为默认参数的坑）
        self.items = items if items is not None else []
        self.cb.addItems(self.items)

        # 信号连接方式保持不变
        # 注意：确保 parent 确实有名为 combo_selection_changed 的方法
        if parent and hasattr(parent, 'combo_selection_changed'):
            self.cb.currentIndexChanged.connect(parent.combo_selection_changed)

        layout.addWidget(self.cb)
        
        # 移除边距（可选，通常作为子组件时会让布局更紧凑）
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def update_items(self, items):
        self.items = items
        # 阻止信号发送，避免清空时触发意外的 index 变化逻辑（视业务需求而定）
        self.cb.blockSignals(True)
        self.cb.clear()
        self.cb.addItems(self.items)
        self.cb.blockSignals(False)