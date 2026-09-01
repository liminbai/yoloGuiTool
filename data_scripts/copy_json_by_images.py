#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量拷贝与图片同名的 JSON 标注文件。
用法：
    python copy_json_by_images.py --img-dir /path/to/images --json-src /path/to/json/source --json-dst /path/to/json/dest
可选：
    --recursive    递归处理图片子目录
    --ext .jpg .png  指定图片扩展名（默认 .jpg .jpeg .png .bmp）
    --overwrite    覆盖目标目录中已存在的 JSON 文件（默认跳过）
    --dry-run      预览模式，不实际拷贝
"""

import argparse
import shutil
from pathlib import Path
from typing import List, Set

# ==================== 日志 ====================
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")

# ==================== 核心功能 ====================

def collect_image_files(img_dir: Path, extensions: List[str], recursive: bool) -> List[Path]:
    """收集图片目录中的所有图片文件路径。"""
    if recursive:
        # 递归遍历所有子目录
        pattern = "**/*"
        all_files = list(img_dir.glob(pattern))
    else:
        all_files = list(img_dir.glob("*"))

    # 过滤扩展名（不区分大小写）
    ext_set = {ext.lower() for ext in extensions}
    img_files = [p for p in all_files if p.suffix.lower() in ext_set and p.is_file()]
    return img_files

def copy_matching_json(
    img_files: List[Path],
    json_src_dir: Path,
    json_dst_dir: Path,
    overwrite: bool = False,
    dry_run: bool = False
) -> int:
    """
    对于每张图片，在 json_src_dir 中查找同名的 .json 文件，
    如果找到则拷贝到 json_dst_dir。
    返回成功拷贝的数量。
    """
    json_src_dir = Path(json_src_dir)
    json_dst_dir = Path(json_dst_dir)
    json_dst_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    missing_count = 0
    skipped_count = 0

    for img_path in img_files:
        base_name = img_path.stem  # 不含扩展名的文件名
        json_name = f"{base_name}.json"
        src_json = json_src_dir / json_name
        dst_json = json_dst_dir / json_name

        if not src_json.exists():
            log_warn(f"未找到对应的 JSON: {src_json}")
            missing_count += 1
            continue

        # 检查目标是否已存在
        if dst_json.exists() and not overwrite:
            log_warn(f"目标已存在，跳过: {dst_json}")
            skipped_count += 1
            continue

        if not dry_run:
            shutil.copy2(src_json, dst_json)  # copy2 保留元数据
        log_info(f"拷贝: {src_json} -> {dst_json}")
        copied_count += 1

    if not dry_run:
        log_info(f"完成！成功拷贝 {copied_count} 个 JSON，跳过 {skipped_count} 个，缺失 {missing_count} 个。")
    else:
        log_info(f"预览：将拷贝 {copied_count} 个 JSON，跳过 {skipped_count} 个，缺失 {missing_count} 个。")

    return copied_count

# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="根据图片文件名批量拷贝对应的 JSON 标注文件"
    )
    parser.add_argument(
        "--img-dir", required=True,
        help="图片目录路径"
    )
    parser.add_argument(
        "--json-src", required=True,
        help="源 JSON 目录路径"
    )
    parser.add_argument(
        "--json-dst", required=True,
        help="目标 JSON 目录路径（将自动创建）"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="递归处理图片子目录（JSON 源目录也需有对应子目录结构？目前仅按文件名匹配，不保留子目录）"
    )
    parser.add_argument(
        "--ext", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="图片扩展名列表，默认 .jpg .jpeg .png .bmp"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="覆盖目标目录中已存在的 JSON 文件（默认跳过）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式：仅显示将要执行的操作，不实际拷贝"
    )

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists() or not img_dir.is_dir():
        log_error(f"图片目录不存在或不是目录: {img_dir}")
        return 1

    json_src = Path(args.json_src)
    if not json_src.exists() or not json_src.is_dir():
        log_error(f"源 JSON 目录不存在或不是目录: {json_src}")
        return 1

    # 收集图片文件
    img_files = collect_image_files(img_dir, args.ext, args.recursive)
    if not img_files:
        log_warn(f"在 {img_dir} 中未找到任何图片文件（扩展名: {args.ext}）")
        return 0

    log_info(f"找到 {len(img_files)} 张图片")

    # 执行拷贝
    copy_matching_json(
        img_files=img_files,
        json_src_dir=json_src,
        json_dst_dir=args.json_dst,
        overwrite=args.overwrite,
        dry_run=args.dry_run
    )

    return 0

if __name__ == "__main__":
    exit(main())