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
            print("循环打印文件路径: ", path)
            self.__load_bundle(path)

    @classmethod
    def get_bundle(cls, locale_str=None):
        if locale_str is None:
            try:
                # locale.getdefaultlocale() 在 Python 3.11+ 中被弃用
                # 推荐使用 locale.getlocale() 或直接处理环境变量
                loc = locale.getlocale()[0]
                locale_str = loc if loc else os.getenv('LANG', 'en').split('.')[0]
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

        print("文件系统基础路径: ", fs_base_path)

        result_paths.append(fs_base_path)
        
        if locale_str is not None:
            tags = re.split('[^a-zA-Z]', locale_str)
            for tag in tags:
                if tag: # 确保 tag 不为空
                    last_path = result_paths[-1]
                    result_paths.append(last_path + '-' + tag)

        return result_paths

    def __load_bundle(self, path_prefix):
        """
        从给定的路径前缀开始，按优先级查找并加载一个配置文件。
        优先级：path_prefix-zh_CN > path_prefix-zh > path_prefix
        仅加载找到的第一个存在且有效的文件。
        """
        PROP_SEPARATOR = '='
        
        # 1. 根据locale构建的tags，生成一个从具体到通用的查找列表
        # 例如：对于 path_prefix=":/strings", tags=['zh', 'CN']，
        # 会生成 [':/strings-zh-CN', ':/strings-zh', ':/strings']
        lookup_paths = []
        import re
        # 假设你的文件名使用连字符 '-' 连接区域标签，如 strings-zh-CN
        # 如果实际使用下划线 '_'，请将 '-' 替换为 '_'
        if hasattr(self, '_locale_tags'): # 需要先在 __init__ 中生成并保存tags
            current_path = path_prefix
            for tag in self._locale_tags:
                current_path += '-' + tag
                lookup_paths.insert(0, current_path) # 插入到开头，保证顺序
        lookup_paths.append(path_prefix) # 最后添加最通用的路径

        # 2. 按优先级查找并加载
        for load_path in lookup_paths:
            # 首先尝试Qt资源系统
            f = QFile(load_path)
            if f.exists():
                print(f"[Info] 尝试从Qt资源加载: {load_path}")
                if f.open(QIODevice.OpenModeFlag.ReadOnly | QIODevice.OpenModeFlag.Text):
                    text_stream = QTextStream(f)
                    # text_stream.setEncoding(QStringConverter.Encoding.Utf8) # PySide6 如果需要
                    while not text_stream.atEnd():
                        line = ustr(text_stream.readLine())
                        if PROP_SEPARATOR in line:
                            key_value = line.split(PROP_SEPARATOR)
                            key = key_value[0].strip()
                            value = PROP_SEPARATOR.join(key_value[1:]).strip().strip('"')
                            self.id_to_message[key] = value
                    f.close()
                    print(f"[Info] 成功从Qt资源加载: {load_path}")
                    return True # 成功加载一个文件后立即返回
            
            # 其次，如果Qt资源不存在，尝试文件系统（仅当路径不是资源路径时）
            elif not load_path.startswith(":/"):
                fs_path = load_path + '.properties' # 补全文件后缀
                if os.path.exists(fs_path):
                    print(f"[Info] 尝试从文件系统加载: {fs_path}")
                    try:
                        with open(fs_path, 'r', encoding='utf-8') as file:
                            for line in file:
                                if PROP_SEPARATOR in line:
                                    key_value = line.split(PROP_SEPARATOR)
                                    key = key_value[0].strip()
                                    value = PROP_SEPARATOR.join(key_value[1:]).strip().strip('"')
                                    self.id_to_message[key] = value
                        print(f"[Info] 成功从文件系统加载: {fs_path}")
                        return True # 成功加载一个文件后立即返回
                    except Exception as e:
                        print(f'[Warning] 加载文件 {fs_path} 失败: {e}')
                        continue # 加载失败，继续尝试下一个更通用的路径
        
        # 3. 所有路径都未找到
        print(f"[Warning] 未找到任何配置文件，路径前缀: {path_prefix}")
        return False