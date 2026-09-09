#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量重命名 X-AnyLabeling 数据集中的图片和 JSON 标注文件（支持分离目录）。
功能：
- 图片和 JSON 可位于不同目录
- 通过通配符（--find）批量查找/筛选文件名，不提供 --replace 时仅作为筛选条件
- 在筛选基础上可自由叠加：通配符替换（--replace）、前缀/后缀（--prefix/--suffix）、
  重新设定数字序列（--start-num/--seq-width，对命中的文件统一重排为 前缀+序号+后缀）
- 自动更新 JSON 中的 "imagePath" 字段
- 支持预览模式（--dry-run）
- 支持递归处理图片子目录（JSON 仅在根目录查找）
"""

import os
import re
import json
import argparse
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

# ==================== 日志 ====================
def log_info(msg: str) -> None:
    print(f"[INFO] {msg}")

def log_warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def log_error(msg: str) -> None:
    print(f"[ERROR] {msg}")

# ==================== 文件名替换函数 ====================

def wildcard_to_regex(pattern: str) -> str:
    """
    将通配符模式（* 和 ?）转换为正则表达式。
    - * 匹配任意字符（包括空）
    - ? 匹配单个任意字符
    返回的正则表达式用于 re.sub，且包含捕获组。
    示例：
        "IMG_*" -> "^IMG_(.*)$"
        "img_??" -> "^img_(..)$"
    """
    # 转义除 * 和 ? 之外的正则特殊字符
    escaped = re.escape(pattern)
    # 将转义后的 \* 恢复为 (.*)，\? 恢复为 (.)
    escaped = escaped.replace(r'\*', '(.*)')
    escaped = escaped.replace(r'\?', '(.)')
    # 确保完全匹配文件名（不含扩展名）
    return '^' + escaped + '$'

def apply_pattern_replace(stem: str, find_pattern: str, replace_str: str) -> str:
    """
    对文件名主体应用通配符替换。
    - find_pattern 包含 * 和 ?，将被自动转换为正则
    - replace_str 可以包含 \1, \2 等引用捕获组
    返回替换后的新 stem。
    """
    regex = wildcard_to_regex(find_pattern)
    # 使用 re.sub，注意 re.sub 支持 \1 引用
    new_stem = re.sub(regex, replace_str, stem)
    return new_stem

# ==================== 重命名核心函数 ====================

def rename_dataset(
    img_dir: Path,
    json_dir: Path,
    recursive: bool = False,
    find: Optional[str] = None,
    replace: Optional[str] = None,
    prefix: str = '',
    suffix: str = '',
    start_num: Optional[int] = None,
    seq_width: int = 4,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """
    执行重命名（各步骤可自由组合）。

    - find: 通配符筛选条件（* 匹配任意、? 匹配单字符）。提供时仅保留文件名
      （不含扩展名）能匹配上的图片；不提供则处理目录内全部图片。
    - replace: 与 find 搭配，对命中的文件名主体做通配符替换（可含 \\1 引用捕获组）。
      不提供 replace 时，find 只作为“筛选条件”，不改名。
    - prefix/suffix: 叠加在最终文件名（或替换结果）前后。
    - start_num/seq_width: 若给定，忽略原名，把被处理到的文件统一重排为
      prefix + 序号 + suffix，实现“重新设定数字序列”。
    - 返回成功重命名的文件对数。
    """
    # 收集图片文件
    if recursive:
        img_files = list(img_dir.rglob('*'))
    else:
        img_files = list(img_dir.glob('*'))
    # 过滤图片扩展名（常见格式）
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    img_files = [p for p in img_files if p.suffix.lower() in img_exts and p.is_file()]
    img_files.sort(key=lambda p: p.name)

    if not img_files:
        log_warn(f"在 {img_dir} 中未找到图片文件")
        return 0

    # ---- 第 1 步：通配符批量查找/筛选（可选）----
    use_filter = (find is not None)
    if use_filter:
        regex = re.compile(wildcard_to_regex(find))
        total_before = len(img_files)
        img_files = [p for p in img_files if regex.match(p.stem)]
        log_info(f"通配符 '{find}' 匹配到 {len(img_files)}/{total_before} 张图片")
        if not img_files:
            log_warn("没有文件匹配该通配符模式，无需处理")
            return 0

    log_info(f"将处理 {len(img_files)} 张图片")

    renumber = (start_num is not None)
    use_replace = (replace is not None)
    if renumber:
        log_info(
            f"使用数字序列：起始编号 {start_num}，位数 {seq_width}"
            + (f"，前缀/后缀: '{prefix}'/'{suffix}'" if (prefix or suffix) else "")
        )
    elif use_replace:
        log_info(
            f"使用通配符替换: '{find}' -> '{replace}'"
            + (f"，叠加前缀/后缀: '{prefix}'/'{suffix}'" if (prefix or suffix) else "")
        )
    else:
        log_info(f"使用前缀/后缀模式: 前缀='{prefix}', 后缀='{suffix}'")

    success_count = 0
    seq = start_num

    for img_path in img_files:
        # 对应的 JSON 文件（在 json_dir 根目录下查找同名 .json）
        base_name = img_path.stem
        json_name = f"{base_name}.json"
        json_path = json_dir / json_name

        if not json_path.exists():
            log_warn(f"未找到对应的 JSON: {json_path}")
            continue

        # ---- 第 2 步：生成新文件名（筛选 → 替换 → 加前缀/后缀 / 重排编号）----
        old_stem = img_path.stem
        old_ext = img_path.suffix

        if renumber:
            # 数字序列模式：忽略原名，统一重排
            new_stem = f"{prefix}{str(seq).zfill(seq_width)}{suffix}"
            seq += 1
        else:
            # 先做通配符替换（如提供），再叠加前缀/后缀
            base = apply_pattern_replace(old_stem, find, replace) if use_replace else old_stem
            new_stem = f"{prefix}{base}{suffix}"

        # 如果结果和原名一样，跳过
        if new_stem == old_stem:
            log_warn(f"文件名未发生变化，跳过: {img_path.name}")
            continue

        new_img_name = new_stem + old_ext
        new_json_name = new_stem + ".json"

        new_img_path = img_path.parent / new_img_name
        new_json_path = json_path.parent / new_json_name

        # 检查目标文件是否已存在
        if (new_img_path.exists() or new_json_path.exists()) and not force:
            log_warn(f"目标文件已存在，跳过: {new_img_name} (使用 --force 覆盖)")
            continue

        # 执行重命名
        if not dry_run:
            # 重命名图片
            img_path.rename(new_img_path)
            # 重命名 JSON
            json_path.rename(new_json_path)
            # 更新 JSON 内部的 imagePath 字段
            update_json_imagepath(new_json_path, new_img_name, dry_run=False)
        else:
            log_info(f"[预览] 重命名: {img_path.name} -> {new_img_name}")
            log_info(f"[预览] 重命名: {json_path.name} -> {new_json_name}")

        success_count += 1
        log_info(f"重命名成功: {img_path.name} -> {new_img_name}")

    if not dry_run:
        log_info(f"完成！成功重命名 {success_count} 对文件。")
    else:
        log_info(f"预览完成，将重命名 {success_count} 对文件。")

    return success_count

def update_json_imagepath(json_path: Path, new_image_name: str, dry_run: bool = False) -> bool:
    """更新 JSON 文件中的 imagePath 字段为新文件名。"""
    if dry_run:
        return True
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data['imagePath'] = new_image_name
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        log_error(f"更新 JSON 失败 {json_path}: {e}")
        return False

# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="批量重命名 X-AnyLabeling 数据集的图片和 JSON（支持分离目录）"
    )
    # 必选参数
    parser.add_argument(
        "--img-dir", required=True,
        help="图片目录路径"
    )
    parser.add_argument(
        "--json-dir", required=True,
        help="JSON 标注文件目录路径"
    )

    # 重命名操作参数（各项可自由组合）
    parser.add_argument(
        "--find", type=str,
        help="通配符筛选/替换模式（* 匹配任意、? 匹配单字符，例如 'IMG_*'）。"
             "仅文件名（不含扩展名）能匹配上的图片才会被处理；不提供则处理全部图片"
    )
    parser.add_argument(
        "--replace", type=str,
        help="替换字符串（与 --find 搭配），可使用 \\1 引用捕获组。"
             "不提供 --replace 时，--find 仅作为筛选条件，不改名"
    )
    parser.add_argument(
        "--prefix", type=str, default="",
        help="添加前缀（可与 --find/--replace/--start-num 组合）"
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="添加后缀（在扩展名前插入，可与 --find/--replace/--start-num 组合）"
    )
    parser.add_argument(
        "--start-num", type=int,
        help="启用数字序列并指定起始编号（如 1）：忽略原名，将被处理的文件统一重排为"
             "前缀+序号+后缀，可与 --find 组合（仅对筛选命中的文件重新编号）"
    )
    parser.add_argument(
        "--seq-width", type=int, default=4,
        help="数字序列位数，默认 4"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="递归遍历图片子目录（JSON 仅在根目录查找）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="覆盖已存在的目标文件（默认跳过）"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="预览模式：仅显示将要执行的操作，不实际修改"
    )

    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists() or not img_dir.is_dir():
        log_error(f"图片目录不存在或不是目录: {img_dir}")
        return 1

    json_dir = Path(args.json_dir)
    if not json_dir.exists() or not json_dir.is_dir():
        log_error(f"JSON 目录不存在或不是目录: {json_dir}")
        return 1

    # 检查参数有效性
    if args.replace is not None and args.find is None:
        log_error("使用 --replace 时必须同时提供 --find")
        return 1

    # 是否指定了任何有效操作（--find 单独出现即可作为筛选条件）
    has_action = (
        args.find is not None
        or args.replace is not None
        or bool(args.prefix)
        or bool(args.suffix)
        or args.start_num is not None
    )
    if not has_action:
        log_error("未指定任何重命名操作，请提供 --find（可带 --replace）或 --prefix/--suffix/--start-num")
        return 1

    # 执行重命名
    rename_dataset(
        img_dir=img_dir,
        json_dir=json_dir,
        recursive=args.recursive,
        find=args.find,
        replace=args.replace,
        prefix=args.prefix,
        suffix=args.suffix,
        start_num=args.start_num,
        seq_width=args.seq_width,
        dry_run=args.dry_run,
        force=args.force,
    )

    return 0

if __name__ == "__main__":
    exit(main())