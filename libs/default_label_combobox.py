import sys
from PySide6.QtWidgets import QWidget, QHBoxLayout, QComboBox

class DefaultLabelComboBox(QWidget):
    def __init__(self, parent=None, items=None):
        # Python 3 推荐的简化写法
        super().__init__(parent)

        layout = QHBoxLayout()
        self.cb = QComboBox()
        
        # 修正默认参数陷阱：避免直接在函数签名中使用 []
        self.items = items if items is not None else []
        self.cb.addItems(self.items)

        # 信号连接
        # 注意：PySide6 中 parent 必须在连接前已定义目标方法
        if parent and hasattr(parent, 'default_label_combo_selection_changed'):
            self.cb.currentIndexChanged.connect(parent.default_label_combo_selection_changed)

        layout.addWidget(self.cb)
        
        # 移除布局外边距，使组件作为子部件时更紧凑
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)