import sys
from PySide6.QtGui import QColor
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QColorDialog, QDialogButtonBox, QVBoxLayout

# 在 PySide6 中，建议直接引用完整的枚举路径
BB = QDialogButtonBox

class ColorDialog(QColorDialog):

    def __init__(self, parent=None):
        super(ColorDialog, self).__init__(parent)
        
        # 1. 选项设置：PySide6 使用完整的枚举路径
        self.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel)
        self.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)
        
        self.default = None
        
        # 2. 布局访问：获取 Dialog 中的 ButtonBox
        # 技巧：在不同版本的 Qt 中，内部布局结构可能略有不同
        # 这里保留 itemAt(1) 的逻辑，但在 PySide6 中通常更稳健的做法是 findChild
        self.bb = self.findChild(QDialogButtonBox)
        
        if self.bb:
            # 3. 添加按钮：使用 StandardButton 枚举
            self.bb.addButton(QDialogButtonBox.StandardButton.RestoreDefaults)
            self.bb.clicked.connect(self.check_restore)

    def getColor(self, value=None, title=None, default=None):
        self.default = default
        if title:
            self.setWindowTitle(title)
        if value:
            # 如果传入的是 QColor 对象直接设置，如果是其他格式需转换
            self.setCurrentColor(QColor(value))
            
        # 4. 运行对话框：PySide6 中 exec_() 已更名为 exec()，但为了兼容性 exec_() 仍可用
        return self.currentColor() if self.exec() else None

    def check_restore(self, button):
        # 5. 角色检查：PySide6 的位运算和枚举路径
        if self.bb.buttonRole(button) == QDialogButtonBox.ButtonRole.ResetRole and self.default:
            self.setCurrentColor(QColor(self.default))