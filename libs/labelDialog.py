import sys
import os
import codecs
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

    def __init__(self, text="Enter object label", parent=None, list_item=None, predef_classes_file=None):
        super().__init__(parent)
        # Store reference to predefined classes file for persistence
        self.predef_classes_file = predef_classes_file

        self.edit = QLineEdit()
        self.edit.setText(text)
        self.edit.setValidator(label_validator())
        self.edit.editingFinished.connect(self.post_process)

        model = QStringListModel()
        # Keep a reference to the original list so we can modify it in-place
        self.list_ref = list_item if list_item is not None else []
        model.setStringList(self.list_ref)
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
            # Allow selecting multiple items for deletion
            self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            for item in list_item:
                self.list_widget.addItem(item)
            self.list_widget.itemClicked.connect(self.list_item_click)
            self.list_widget.itemDoubleClicked.connect(self.list_item_double_click)
            # Add Add/Delete buttons under the list
            btn_hbox = QHBoxLayout()
            self.add_button = QPushButton('Add', self)
            self.delete_button = QPushButton('Delete Selected', self)
            btn_hbox.addWidget(self.add_button)
            btn_hbox.addWidget(self.delete_button)
            self.add_button.clicked.connect(self.add_label_from_edit)
            self.delete_button.clicked.connect(self.delete_selected_labels)

            layout.addWidget(self.list_widget)
            layout.addLayout(btn_hbox)

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

    def add_label_from_edit(self):
        """Add the current edit text to the list if non-empty and not present."""
        text = trimmed(self.edit.text())
        if not text:
            return
        # Avoid duplicates
        if text in self.list_ref:
            # select the existing item
            items = self.list_widget.findItems(text, Qt.MatchFlag.MatchExactly)
            if items:
                self.list_widget.setCurrentItem(items[0])
            return
        # Append to underlying list and UI
        self.list_ref.append(text)
        self.list_widget.addItem(text)
        # Persist to predefined_classes.txt
        self._save_predefined_classes()
        # Update completer model
        model = QStringListModel()
        model.setStringList(self.list_ref)
        completer = QCompleter()
        completer.setModel(model)
        self.edit.setCompleter(completer)

    def delete_selected_labels(self):
        """Delete selected labels from the list and underlying list_ref."""
        items = self.list_widget.selectedItems()
        if not items:
            return
        for it in items:
            text = trimmed(it.text())
            # Remove from list_widget
            row = self.list_widget.row(it)
            self.list_widget.takeItem(row)
            # Remove from underlying list if present
            try:
                while text in self.list_ref:
                    self.list_ref.remove(text)
            except Exception:
                pass
        # Persist to predefined_classes.txt
        self._save_predefined_classes()
        # Update completer model
        model = QStringListModel()
        model.setStringList(self.list_ref)
        completer = QCompleter()
        completer.setModel(model)
        self.edit.setCompleter(completer)

    def _save_predefined_classes(self):
        """Save the current label list to predefined_classes.txt."""
        if not self.predef_classes_file:
            return
        try:
            # Create directory if it doesn't exist
            dirname = os.path.dirname(self.predef_classes_file)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            # Write labels to file, one per line
            with codecs.open(self.predef_classes_file, 'w', 'utf8') as f:
                for label in self.list_ref:
                    f.write(label + '\n')
        except Exception as e:
            print(f"Failed to save predefined_classes.txt: {e}")