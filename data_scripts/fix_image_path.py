#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

def process_json(file_path):
    """处理单个 JSON 文件，修改 imagePath 字段中的后缀 .jpg -> .jpeg"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 读取失败 {file_path}: {e}")
        return

    modified = False
    if 'imagePath' in data and isinstance(data['imagePath'], str):
        old = data['imagePath']
        # 不区分大小写匹配 .jpg 结尾
        if old.lower().endswith('.jpg'):
            # 替换后缀（保留原大小写，仅改扩展名）
            new = old[:-4] + '.jpeg'
            data['imagePath'] = new
            modified = True
            print(f"✅ {file_path}: '{old}' -> '{new}'")

    if modified:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 写入失败 {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="将指定目录下所有 JSON 文件中的 imagePath 字段的 .jpg 后缀改为 .jpeg"
    )
    parser.add_argument(
        'directory', nargs='?', default='.',
        help='要扫描的目录路径（默认当前目录）'
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ 错误：'{args.directory}' 不是有效的目录")
        return

    # 递归收集所有 .json 文件
    json_files = []
    for root, _, files in os.walk(args.directory):
        for f in files:
            if f.lower().endswith('.json'):
                json_files.append(os.path.join(root, f))

    if not json_files:
        print(f"📭 在 '{args.directory}' 下未找到 JSON 文件")
        return

    print(f"📂 找到 {len(json_files)} 个 JSON 文件，开始处理...\n")
    for fp in json_files:
        process_json(fp)

    print("\n🎉 处理完成！")

if __name__ == '__main__':
    main()