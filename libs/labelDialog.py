import sys
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *

# 假设这些本地库已经存在
# from libs.utils import new_icon, label_validator, trimmed

# 如果没有对应的本地库，为了代码可运行，这里提供简单的模拟逻辑
try:
    from libs.utils import new_icon, label_validator, trimmed
except ImportError:
    def new_icon(name): return QIcon()
    def label_validator(): return None
    def trimmed(text): return text.strip()

BB = QDialogButtonBox

class LabelDialog(QDialog):

    def __init__(self, text="Enter object label", parent=None, list_item=None):
        super().__init__(parent)

        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setValidator(label_validator())
        self.edit.editingFinished.connect(self.post_process)

        model = QStringListModel()
        model.setStringList(list_item if list_item else [])
        completer = QCompleter()
        completer.setModel(model)
        self.edit.setCompleter(completer)

        # PySide6 枚举引用调整
        self.button_box = bb = BB(BB.StandardButton.Ok | BB.StandardButton.Cancel, 
                                  Qt.Orientation.Horizontal, self)
        
        bb.button(BB.StandardButton.Ok).setIcon(new_icon('done'))
        bb.button(BB.StandardButton.Cancel).setIcon(new_icon('undo'))
        bb.accepted.connect(self.validate)
        bb.rejected.connect(self.reject)

        layout = QVBoxLayout()
        # 对齐方式调整
        layout.addWidget(bb, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.edit)

        if list_item is not None and len(list_item) > 0:
            self.list_widget = QListWidget(self)
            for item in list_item:
                self.list_widget.addItem(item)
            self.list_widget.itemClicked.connect(self.list_item_click)
            self.list_widget.itemDoubleClicked.connect(self.list_item_double_click)
            layout.addWidget(self.list_widget)

        self.setLayout(layout)

    def validate(self):
        if trimmed(self.edit.text()):
            self.accept()

    def post_process(self):
        self.edit.setText(trimmed(self.edit.text()))

    def pop_up(self, text='', move=True):
        self.edit.setText(text)
        self.edit.setSelection(0, len(text))
        # 枚举引用调整
        self.edit.setFocus(Qt.FocusReason.PopupFocusReason)
        
        if move:
            cursor_pos = QCursor.pos()
            btn = self.button_box.buttons()[0]
            self.adjustSize()
            btn.adjustSize()
            
            # mapToGlobal 在 PySide6 中返回 QPoint
            offset = btn.mapToGlobal(btn.pos()) - self.pos()
            offset += QPoint(btn.size().width() // 4, btn.size().height() // 2)
            cursor_pos.setX(max(0, cursor_pos.x() - offset.x()))
            cursor_pos.setY(max(0, cursor_pos.y() - offset.y()))

            if self.parentWidget():
                parent_bottom_right = self.parentWidget().geometry()
                max_x = parent_bottom_right.x() + parent_bottom_right.width() - self.sizeHint().width()
                max_y = parent_bottom_right.y() + parent_bottom_right.height() - self.sizeHint().height()
                max_global = self.parentWidget().mapToGlobal(QPoint(max_x, max_y))
                if cursor_pos.x() > max_global.x():
                    cursor_pos.setX(max_global.x())
                if cursor_pos.y() > max_global.y():
                    cursor_pos.setY(max_global.y())
            
            self.move(cursor_pos)
            
        # PySide6 建议使用 exec() 代替 exec_()
        return trimmed(self.edit.text()) if self.exec() else None

    def list_item_click(self, t_qlist_widget_item):
        text = trimmed(t_qlist_widget_item.text())
        self.edit.setText(text)

    def list_item_double_click(self, t_qlist_widget_item):
        self.list_item_click(t_qlist_widget_item)
        self.validate()