from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *

class ToolBar(QToolBar):

    def __init__(self, title):
        # Python 3 中 super() 不需要传参数
        super().__init__(title)
        
        layout = self.layout()
        m = (0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setContentsMargins(*m)
        self.setContentsMargins(*m)
        
        # PySide6 枚举：使用完整的 WindowType 路径
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)

    def addAction(self, action):
        # 检查是否为 QWidgetAction
        if isinstance(action, QWidgetAction):
            return super().addAction(action)
        
        btn = ToolButton()
        btn.setDefaultAction(action)
        btn.setToolButtonStyle(self.toolButtonStyle())
        self.addWidget(btn)


class ToolButton(QToolButton):
    """确保所有按钮具有相同尺寸的 ToolBar 伴生类"""
    minSize = (60, 60)

    def minimumSizeHint(self):
        ms = super().minimumSizeHint()
        w1, h1 = ms.width(), ms.height()
        w2, h2 = self.minSize
        # 更新类静态变量以确保同步尺寸
        ToolButton.minSize = max(w1, w2), max(h1, h2)
        return QSize(*ToolButton.minSize)