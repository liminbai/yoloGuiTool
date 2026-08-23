#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

def find_keys_recursive(data, target_key, current_path="", results=None):
    """递归查找所有键为 target_key 的字段，记录路径和值"""
    if results is None:
        results = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f"{current_path}{k}" if current_path else k
            if k == target_key:
                results.append((new_path, v))
            find_keys_recursive(v, target_key, new_path + ".", results)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{current_path}[{idx}]"
            find_keys_recursive(item, target_key, new_path + ".", results)
    return results

def replace_field_recursive(data, key, old_value, new_value, current_path=""):
    """递归替换所有精确匹配 (key, old_value) 的字段，返回修改次数（大小写敏感）"""
    count = 0
    if isinstance(data, dict):
        if key in data and data[key] == old_value:
            data[key] = new_value
            count += 1
            path_str = f"{current_path}{key}" if current_path else key
            print(f"   ✅ 替换 {path_str} : '{old_value}' -> '{new_value}'")
        for k, v in data.items():
            new_path = f"{current_path}{k}." if current_path else f"{k}."
            count += replace_field_recursive(v, key, old_value, new_value, new_path)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            new_path = f"{current_path}[{idx}]."
            count += replace_field_recursive(item, key, old_value, new_value, new_path)
    return count

def process_json(file_path, fix_image=True, label_old=None, label_new=None,
                 custom_key=None, custom_old=None, custom_new=None, debug=False):
    """处理单个 JSON 文件，返回修改次数"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️ 读取失败 {file_path}: {e}")
        return 0

    # debug 模式：只查找并打印
    if debug:
        target_key = None
        if label_old is not None:
            target_key = 'label'
        elif custom_key is not None:
            target_key = custom_key
        else:
            print(f"ℹ️ debug 模式未指定查找键，请提供 --label-old 或 --key")
            return 0
        print(f"\n🔎 调试模式：查找所有 '{target_key}' 字段 (文件: {os.path.basename(file_path)})")
        results = find_keys_recursive(data, target_key)
        if results:
            for path, val in results:
                print(f"   📍 {path} = {repr(val)}")
        else:
            print(f"   ⚠️ 未找到任何键为 '{target_key}' 的字段")
        return 0

    modified_count = 0

    # 修正 imagePath（仅顶层）
    if fix_image:
        if 'imagePath' in data and isinstance(data['imagePath'], str):
            old = data['imagePath']
            if old.lower().endswith('.jpg'):
                new = old[:-4] + '.jpeg'
                data['imagePath'] = new
                modified_count += 1
                print(f"✅ {file_path}: imagePath '{old}' -> '{new}'")

    # 替换 label（递归，大小写敏感）
    if label_old is not None and label_new is not None:
        print(f"🔍 在 {file_path} 中递归替换 label 字段...")
        cnt = replace_field_recursive(data, 'label', label_old, label_new)
        if cnt == 0:
            print(f"   ⚠️ 未找到任何 label 值精确匹配 '{label_old}' 的字段")
        modified_count += cnt

    # 替换自定义字段（递归，大小写敏感）
    if custom_key is not None and custom_old is not None and custom_new is not None:
        print(f"🔍 在 {file_path} 中递归替换 {custom_key} 字段...")
        cnt = replace_field_recursive(data, custom_key, custom_old, custom_new)
        if cnt == 0:
            print(f"   ⚠️ 未找到任何 {custom_key} 值精确匹配 '{custom_old}' 的字段")
        modified_count += cnt

    if modified_count > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ 写入失败 {file_path}: {e}")
            print(f"   ⚠️ 实际修改未保存，计数可能不准确")

    return modified_count

def main():
    parser = argparse.ArgumentParser(
        description="批量处理 JSON：修正 imagePath 后缀，递归替换指定字段（大小写敏感）"
    )
    parser.add_argument('directory', nargs='?', default='.', help='目录路径')
    parser.add_argument('--no-image-fix', action='store_true', help='禁用 imagePath 后缀修正')
    parser.add_argument('--label-old', help='label 旧值（如 Person）')
    parser.add_argument('--label-new', help='label 新值（如 person）')
    parser.add_argument('--key', help='自定义字段名')
    parser.add_argument('--old-value', help='自定义字段旧值')
    parser.add_argument('--new-value', help='自定义字段新值')
    parser.add_argument('--debug', action='store_true', help='调试模式：只查找不修改')

    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"❌ 错误：'{args.directory}' 不是有效的目录")
        return

    json_files = []
    for root, _, files in os.walk(args.directory):
        for f in files:
            if f.lower().endswith('.json'):
                json_files.append(os.path.join(root, f))

    if not json_files:
        print(f"📭 在 '{args.directory}' 下未找到 JSON 文件")
        return

    print(f"📂 找到 {len(json_files)} 个 JSON 文件，开始处理...\n")

    total_mods = 0
    for fp in json_files:
        total_mods += process_json(
            fp,
            fix_image=not args.no_image_fix,
            label_old=args.label_old,
            label_new=args.label_new,
            custom_key=args.key,
            custom_old=args.old_value,
            custom_new=args.new_value,
            debug=args.debug
        )

    if not args.debug:
        print(f"\n🎉 处理完成！共修改了 {total_mods} 处字段。")
    else:
        print("\n📌 调试模式完成，未修改任何文件。")

if __name__ == '__main__':
    main()