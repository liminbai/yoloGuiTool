from PySide6.QtGui import QColor, QFontMetrics
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QSpinBox, QAbstractSpinBox

class LightWidget(QSpinBox):

    def __init__(self, title, value=50):
        # Python 3 简化了 super() 调用
        super().__init__()
        
        # 1. 枚举路径更新：使用完整路径访问
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setRange(0, 100)
        self.setSuffix(' %')
        self.setValue(value)
        self.setToolTip(title)
        self.setStatusTip(self.toolTip())
        
        # 2. 对齐方式枚举更新
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def minimumSizeHint(self):
        height = super().minimumSizeHint().height()
        fm = QFontMetrics(self.font())
        
        # 3. 重要：Qt6 中 fm.width() 已废弃，改用 horizontalAdvance()
        width = fm.horizontalAdvance(str(self.maximum()))
        
        # 增加一点 padding 确保文字不被遮挡
        return QSize(width + 10, height)

    def color(self):
        if self.value() == 50:
            return None

        # 计算灰度强度
        strength = int(self.value() / 100 * 255 + 0.5)
        return QColor(strength, strength, strength)