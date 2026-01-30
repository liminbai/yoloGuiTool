#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
如果在 resources/strings 文件夹中添加了项，
请在根目录执行 "pyside6-rcc resources.qrc -o resources.py"
并在 libs 目录执行 "pyside6-rcc ../resources.qrc -o resources.py"
"""
import re
import os
import sys
import locale
from libs.ustr import ustr

# 1. 简化的导入逻辑，移除 PyQt4/sip 兼容层
from PySide6.QtCore import QFile, QIODevice, QTextStream

class StringBundle:

    __create_key = object()

    def __init__(self, create_key, locale_str):
        assert(create_key == StringBundle.__create_key), "StringBundle must be created using StringBundle.getBundle"
        self.id_to_message = {}
        paths = self.__create_lookup_fallback_list(locale_str)
        for path in paths:
            self.__load_bundle(path)

    @classmethod
    def get_bundle(cls, locale_str=None):
        if locale_str is None:
            try:
                # locale.getdefaultlocale() 在 Python 3.11+ 中被弃用
                # 推荐使用 locale.getlocale() 或直接处理环境变量
                loc = locale.getlocale()[0]
                locale_str = loc if loc else os.getenv('LANG', 'en')
            except:
                print('Invalid locale')
                locale_str = 'en'

        return StringBundle(cls.__create_key, locale_str)

    def get_string(self, string_id):
        assert(string_id in self.id_to_message), "Missing string id : " + string_id
        return self.id_to_message[string_id]

    def __create_lookup_fallback_list(self, locale_str):
        result_paths = []
        # 优先使用 Qt 资源系统路径
        base_path = ":/strings"
        result_paths.append(base_path)
        
        # 如果 Qt 资源系统加载失败，使用文件系统路径作为后备
        import os
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fs_base_path = os.path.join(script_dir, "resources", "strings", "strings")
        result_paths.append(fs_base_path)
        
        if locale_str is not None:
            tags = re.split('[^a-zA-Z]', locale_str)
            for tag in tags:
                if tag: # 确保 tag 不为空
                    last_path = result_paths[-1]
                    result_paths.append(last_path + '-' + tag)

        return result_paths

    def __load_bundle(self, path):
        PROP_SEPARATOR = '='
        
        # 优先尝试从 Qt 资源系统加载
        f = QFile(path)
        loaded = False
        
        if f.exists():
            # 2. PySide6 枚举使用完整路径
            if f.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text):
                text_stream = QTextStream(f)
                
                # 3. 重要变化：Qt6 中 setCodec 已被移除
                # QTextStream 默认使用 UTF-8。如果需要显式设置：
                # text_stream.setEncoding(QStringConverter.Encoding.Utf8)
                
                while not text_stream.atEnd():
                    line = ustr(text_stream.readLine())
                    if PROP_SEPARATOR in line:
                        key_value = line.split(PROP_SEPARATOR)
                        key = key_value[0].strip()
                        value = PROP_SEPARATOR.join(key_value[1:]).strip().strip('"')
                        self.id_to_message[key] = value

                f.close()
                loaded = True
        
        # 如果 Qt 资源系统加载失败，尝试从文件系统加载（后备方案）
        elif not path.startswith(":/"):
            try:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as file:
                        for line in file:
                            if PROP_SEPARATOR in line:
                                key_value = line.split(PROP_SEPARATOR)
                                key = key_value[0].strip()
                                value = PROP_SEPARATOR.join(key_value[1:]).strip().strip('"')
                                self.id_to_message[key] = value
            except Exception as e:
                print(f'Error loading {path}: {e}')