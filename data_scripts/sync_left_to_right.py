#!/usr/bin/env python3
"""
sync_left_to_right.py

以左侧文件夹为基准，删除右侧文件夹中多余的文件（忽略扩展名）。
即：右侧中与左侧“去扩展名相对路径”不匹配的所有文件将被删除。
用法: python sync_left_to_right.py <左侧文件夹> <右侧文件夹>
"""

import os
import sys
import argparse

def get_file_info(root_dir):
    """
    递归扫描 root_dir，返回:
      - key_set: 所有文件的“去扩展名相对路径”集合
      - key_to_files: 字典，映射 去扩展名相对路径 -> 实际文件路径列表（相对路径）
    """
    root_dir = os.path.abspath(root_dir)
    key_set = set()
    key_to_files = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root_dir)
            # 去掉扩展名（最后一个 . 之后的部分）
            base, _ = os.path.splitext(rel_path)
            key = base  # 去扩展名的相对路径
            key_set.add(key)
            key_to_files.setdefault(key, []).append(rel_path)

    return key_set, key_to_files

def delete_files_by_keys(keys_to_delete, key_to_files, base_dir, dry_run=False):
    """
    删除 base_dir 中，所有去扩展名路径在 keys_to_delete 中的实际文件。
    """
    if not keys_to_delete:
        return 0
    count = 0
    for key in sorted(keys_to_delete):
        rel_paths = key_to_files.get(key, [])
        for rel_path in rel_paths:
            full_path = os.path.join(base_dir, rel_path)
            if dry_run:
                print(f"  [模拟] 将删除: {full_path}")
            else:
                try:
                    os.remove(full_path)
                    print(f"  已删除: {full_path}")
                    count += 1
                except Exception as e:
                    print(f"  删除失败: {full_path} - {e}")
    return count

def main():
    parser = argparse.ArgumentParser(
        description="以左侧文件夹为基准，删除右侧文件夹中多余的文件（忽略扩展名）。"
    )
    parser.add_argument("left_folder", help="左侧基准文件夹路径")
    parser.add_argument("right_folder", help="右侧待清理文件夹路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="模拟运行，只显示将要删除的文件，不实际删除")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="自动确认删除，无需交互")
    args = parser.parse_args()

    left = os.path.abspath(args.left_folder)
    right = os.path.abspath(args.right_folder)

    if not os.path.isdir(left) or not os.path.isdir(right):
        print("错误: 请确保两个路径都是有效目录", file=sys.stderr)
        sys.exit(1)

    print(f"正在扫描左侧基准文件夹 {left} ...")
    left_keys, _ = get_file_info(left)
    print(f"  去扩展名路径数量: {len(left_keys)}")

    print(f"正在扫描右侧待清理文件夹 {right} ...")
    right_keys, right_map = get_file_info(right)
    print(f"  去扩展名路径数量: {len(right_keys)}")

    # 计算右侧中多余的 key（左侧没有的）
    extra_keys = right_keys - left_keys

    if not extra_keys:
        print("右侧文件夹中没有多余的文件（与左侧相比），无需删除。")
        return

    # 统计实际要删除的文件数量
    total = sum(len(right_map[k]) for k in extra_keys)

    print("\n以下文件仅存在于右侧文件夹中（将删除）：")
    for key in sorted(extra_keys):
        for rel in right_map[key]:
            print(f"  {os.path.join(right, rel)}")

    print(f"\n总计 {total} 个文件将被删除（右侧）。左侧文件夹不受影响。")

    if args.dry_run:
        print("\n[模拟模式] 不会实际删除文件。")
        delete_files_by_keys(extra_keys, right_map, right, dry_run=True)
        return

    if not args.yes:
        confirm = input("确认要删除右侧这些文件吗？(y/N): ").strip().lower()
        if confirm != 'y':
            print("操作已取消。")
            return

    print("\n开始删除右侧多余文件...")
    deleted = delete_files_by_keys(extra_keys, right_map, right, dry_run=False)
    print(f"操作完成。共删除了 {deleted} 个文件。")

if __name__ == "__main__":
    main()