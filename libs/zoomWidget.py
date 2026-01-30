from PySide6.QtGui import QFontMetrics
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QSpinBox, QAbstractSpinBox

class ZoomWidget(QSpinBox):

    def __init__(self, value=100):
        # Python 3 推荐的简洁写法
        super().__init__()
        
        # 1. 枚举路径：使用完整的 ButtonSymbols 路径
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        
        self.setRange(1, 500)
        self.setSuffix(' %')
        self.setValue(value)
        self.setToolTip('Zoom Level')
        self.setStatusTip(self.toolTip())
        
        # 2. 对齐方式：使用完整的 AlignmentFlag 路径
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def minimumSizeHint(self):
        # 获取父类建议的高度
        height = super().minimumSizeHint().height()
        
        # 3. 计算文本宽度
        fm = QFontMetrics(self.font())
        # horizontalAdvance 是 Qt6 计算文本占据像素宽度的标准方法
        width = fm.horizontalAdvance(str(self.maximum()))
        
        # 建议：额外增加一些宽度（如 15 像素）以容纳后缀 "%"
        return QSize(width + 15, height)