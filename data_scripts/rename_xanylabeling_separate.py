#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量重命名 X-AnyLabeling 数据集的图片和 JSON 标注文件（图片和 JSON 分别在不同目录）。
功能：
- 图片和 JSON 可分别位于不同目录，但它们的子目录结构应保持一致（或可配置）
- 支持添加前缀、后缀
- 支持按数字序列重命名（可指定起始编号和位数）
- 自动更新 JSON 文件中的 "imagePath" 字段为新图片文件名
- 支持预览模式（--dry-run）
- 支持递归处理子目录（--recursive）
- 支持指定文件扩展名
- 安全：跳过已存在的目标文件，避免覆盖
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict

# ==================== 日志与辅助 ====================

def safe_rename(src: Path, dst: Path, dry_run: bool = False) -> bool:
    """安全重命名，如果目标已存在则跳过并返回 False。"""
    if dst.exists():
        print(f"  ⚠️  跳过: 目标已存在 {dst}")
        return False
    if not dry_run:
        src.rename(dst)
    print(f"  ✅ 重命名: {src.name} -> {dst.name}")
    return True

def update_json_imagepath(json_path: Path, new_image_name: str, dry_run: bool = False) -> bool:
    """更新 JSON 文件中的 imagePath 字段为新的图片文件名。"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ 读取 JSON 失败 {json_path}: {e}")
        return False

    data['imagePath'] = new_image_name

    if not dry_run:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  📝 更新 JSON imagePath: {new_image_name}")
    return True

# ==================== 核心重命名逻辑 ====================

def get_relative_path(file_path: Path, base_dir: Path) -> Path:
    """计算 file_path 相对于 base_dir 的相对路径，如果不包含则返回文件自身。"""
    try:
        return file_path.relative_to(base_dir)
    except ValueError:
        return file_path.name

def rename_dataset_separate(
    img_root: Path,
    json_root: Path,
    prefix: str = '',
    suffix: str = '',
    start_num: Optional[int] = None,
    seq_width: int = 4,
    recursive: bool = False,
    img_exts: List[str] = ['.jpg', '.jpeg', '.png', '.bmp'],
    dry_run: bool = False
) -> None:
    """
    执行批量重命名，图片和 JSON 分别在 img_root 和 json_root 下。
    图片和 JSON 的相对路径结构应一致（即 img_root 下的子目录结构与 json_root 下相同）。
    如果设置了递归，则会处理子目录；否则只处理根目录下的文件。
    """
    if start_num is not None:
        # 使用序列号模式：收集所有图片文件，排序后分配序号
        pattern = "**/*" if recursive else "*"
        img_paths = list(img_root.glob(pattern))
        img_paths = [p for p in img_paths if p.suffix.lower() in [ext.lower() for ext in img_exts]]
        img_paths.sort(key=lambda p: p.name)
        print(f"找到 {len(img_paths)} 张图片，将使用序列号从 {start_num} 开始重命名")
        seq = start_num
        for img_path in img_paths:
            # 计算相对路径（相对于 img_root）
            rel_path = get_relative_path(img_path, img_root)
            # 对应的 JSON 文件路径（在 json_root 中查找相同的相对路径，但扩展名为 .json）
            json_path = json_root / rel_path.with_suffix('.json')
            if not json_path.exists():
                print(f"  ⚠️  跳过: 未找到对应的 JSON 文件 {json_path}")
                seq += 1
                continue

            # 生成新文件名（仅文件名，不含路径）
            new_img_name = f"{prefix}{str(seq).zfill(seq_width)}{suffix}{img_path.suffix}"
            new_json_name = f"{prefix}{str(seq).zfill(seq_width)}{suffix}.json"

            # 目标路径（保持相同相对目录结构）
            new_img_path = img_path.parent / new_img_name
            new_json_path = json_path.parent / new_json_name

            # 检查目标是否存在
            if new_img_path.exists() or new_json_path.exists():
                print(f"  ⚠️  跳过: 目标文件已存在 (img: {new_img_path}, json: {new_json_path})")
                seq += 1
                continue

            # 重命名图片
            if safe_rename(img_path, new_img_path, dry_run):
                # 重命名 JSON
                if safe_rename(json_path, new_json_path, dry_run):
                    # 更新 JSON 内部 imagePath（注意：imagePath 应仅包含文件名，不包含路径）
                    update_json_imagepath(new_json_path, new_img_name, dry_run)
            seq += 1
    else:
        # 无序列号：使用前缀/后缀重命名
        pattern = "**/*" if recursive else "*"
        img_paths = list(img_root.glob(pattern))
        img_paths = [p for p in img_paths if p.suffix.lower() in [ext.lower() for ext in img_exts]]
        print(f"找到 {len(img_paths)} 张图片，将添加前缀/后缀重命名")
        for img_path in img_paths:
            rel_path = get_relative_path(img_path, img_root)
            json_path = json_root / rel_path.with_suffix('.json')
            if not json_path.exists():
                print(f"  ⚠️  跳过: 未找到对应的 JSON 文件 {json_path}")
                continue

            stem = img_path.stem
            new_img_name = f"{prefix}{stem}{suffix}{img_path.suffix}"
            new_json_name = f"{prefix}{stem}{suffix}.json"

            new_img_path = img_path.parent / new_img_name
            new_json_path = json_path.parent / new_json_name

            if new_img_path.exists() or new_json_path.exists():
                print(f"  ⚠️  跳过: 目标文件已存在 (img: {new_img_path}, json: {new_json_path})")
                continue

            if safe_rename(img_path, new_img_path, dry_run):
                if safe_rename(json_path, new_json_path, dry_run):
                    update_json_imagepath(new_json_path, new_img_name, dry_run)

# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="批量重命名 X-AnyLabeling 数据集的图片和 JSON 标注文件（图片和 JSON 可分离目录）。"
    )
    parser.add_argument(
        "--img-dir", required=True,
        help="图片目录路径"
    )
    parser.add_argument(
        "--json-dir", required=True,
        help="JSON 标注文件目录路径（目录结构与图片目录应一致）"
    )
    parser.add_argument(
        "--prefix", default="",
        help="添加前缀"
    )
    parser.add_argument(
        "--suffix", default="",
        help="添加后缀（在扩展名前插入）"
    )
    parser.add_argument(
        "--start-num", type=int, default=None,
        help="启用数字序列，并指定起始编号（如 1）"
    )
    parser.add_argument(
        "--seq-width", type=int, default=4,
        help="序列号位数，默认 4（如 0001）"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="递归处理子目录（图片和 JSON 的子目录结构需对应）"
    )
    parser.add_argument(
        "--img-ext", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="图片扩展名列表，默认 .jpg .jpeg .png .bmp"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式：只显示将要执行的操作，不实际修改文件"
    )
    args = parser.parse_args()

    img_root = Path(args.img_dir)
    json_root = Path(args.json_dir)
    if not img_root.exists() or not img_root.is_dir():
        print(f"错误: 图片目录不存在或不是目录: {img_root}")
        return 1
    if not json_root.exists() or not json_root.is_dir():
        print(f"错误: JSON 目录不存在或不是目录: {json_root}")
        return 1

    if not args.start_num and not args.prefix and not args.suffix:
        print("错误: 必须提供 --start-num 或 --prefix/--suffix 至少一项")
        return 1

    rename_dataset_separate(
        img_root=img_root,
        json_root=json_root,
        prefix=args.prefix,
        suffix=args.suffix,
        start_num=args.start_num,
        seq_width=args.seq_width,
        recursive=args.recursive,
        img_exts=args.img_ext,
        dry_run=args.dry_run
    )

    if args.dry_run:
        print("\n预览完成。若需实际执行，请去掉 --dry-run 参数。")
    else:
        print("\n重命名完成！")

    return 0

if __name__ == "__main__":
    exit(main())