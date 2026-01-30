#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from PySide6.QtGui import *
from PySide6.QtCore import *
from PySide6.QtWidgets import *

# 在 PySide6 中，通常不再需要处理 QVariant 兼容性问题
# 也不再需要使用 sip 模块

class HashableQListWidgetItem(QListWidgetItem):

    def __init__(self, *args):
        # Python 3 推荐的简化 super() 写法
        super().__init__(*args)

    def __hash__(self):
        # 保持原有逻辑：使用对象的内存 ID 作为哈希值
        # 这确保了该类实例可以作为字典的键 (dict keys) 或存入集合 (set)
        return hash(id(self))