from math import sqrt
from libs.ustr import ustr
import hashlib
import re
import sys

# 修改导入路径为 PySide6
from PySide6.QtGui import QIcon, QAction, QColor, QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression, Qt
from PySide6.QtWidgets import QPushButton, QMenu

def new_icon(icon):
    return QIcon(':/' + icon)

def new_button(text, icon=None, slot=None):
    b = QPushButton(text)
    if icon is not None:
        b.setIcon(new_icon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b

def new_action(parent, text, slot=None, shortcut=None, icon=None,
               tip=None, checkable=False, enabled=True):
    """创建新动作并分配回调、快捷键等"""
    # PySide6 中 QAction 位于 QtGui 模块
    a = QAction(text, parent)
    if icon is not None:
        a.setIcon(new_icon(icon))
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    return a

def add_actions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)

def label_validator():
    # 重要：Qt6 使用 QRegularExpression 代替 QRegExp
    # r'^[^ \t].+' 表示匹配开头不是空格或制表符的字符串
    return QRegularExpressionValidator(QRegularExpression(r'^[^ \t].+'), None)

class Struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def distance(p):
    return sqrt(p.x() * p.x() + p.y() * p.y())

def format_shortcut(text):
    mod, key = text.split('+', 1)
    return f'<b>{mod}</b>+<b>{key}</b>'

def generate_color_by_text(text):
    s = ustr(text)
    hash_code = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16)
    r = int((hash_code / 255) % 255)
    g = int((hash_code / 65025) % 255)
    b = int((hash_code / 16581375) % 255)
    return QColor(r, g, b, 100)

def have_qstring():
    """PySide6/Python3 环境下不再需要 QString 封装"""
    return False

def util_qt_strlistclass():
    """PySide6 直接使用 Python 原生 list"""
    return list

def natural_sort(list_to_sort, key=lambda s: s):
    """自然排序算法"""
    def get_alphanum_key_func(k):
        convert = lambda text: int(text) if text.isdigit() else text
        return lambda s: [convert(c) for c in re.split('([0-9]+)', k(s))]
    sort_key = get_alphanum_key_func(key)
    list_to_sort.sort(key=sort_key)

def trimmed(text):
    """Qt6/Python3 环境下直接使用 Python 的 strip()"""
    return text.strip()