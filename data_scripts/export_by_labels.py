#!/usr/bin/env python3
"""
按标签筛选导出 X-AnyLabeling (LabelMe 格式) 数据集
支持图片和 JSON 文件分别在各自的目录，导出时也分别存放。
用法：
    python export_by_labels_separate.py \
        --src-json /path/to/jsons \
        --src-images /path/to/images \
        --dst-json /path/to/output/jsons \
        --dst-images /path/to/output/images \
        --labels person,car,bus
"""

import os
import json
import shutil
import argparse
from pathlib import Path

def find_image_for_json(json_path, src_images_dir):
    """
    根据 JSON 文件查找对应的图片文件。
    优先使用 JSON 内的 imagePath 字段（相对于 JSON 目录或绝对路径），
    否则尝试在 src_images_dir 中查找同名文件（支持常见扩展名）。
    """
    # 读取 JSON 获取 imagePath
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_path_str = data.get("imagePath")
    
    if image_path_str:
        # 尝试多种路径解析
        # 1. 绝对路径
        abs_path = Path(image_path_str)
        if abs_path.is_absolute():
            if abs_path.exists():
                return abs_path
        # 2. 相对于 JSON 文件的路径
        rel_to_json = json_path.parent / image_path_str
        if rel_to_json.exists():
            return rel_to_json
        # 3. 相对于 src_images_dir 的路径
        rel_to_img = src_images_dir / image_path_str
        if rel_to_img.exists():
            return rel_to_img
        # 4. 仅文件名，拼接 src_images_dir
        base_name = Path(image_path_str).name
        candidate = src_images_dir / base_name
        if candidate.exists():
            return candidate
    
    # 如果 imagePath 无效，尝试按文件名匹配（JSON 名 + 常见扩展名）
    base = json_path.stem
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.PNG']:
        candidate = src_images_dir / f"{base}{ext}"
        if candidate.exists():
            return candidate
    
    return None

def export_dataset(json_src_dir, img_src_dir, json_dst_dir, img_dst_dir, labels, copy_others=False):
    """
    主导出函数
    """
    json_src = Path(json_src_dir)
    img_src = Path(img_src_dir)
    json_dst = Path(json_dst_dir)
    img_dst = Path(img_dst_dir)

    labels_set = set(labels)

    # 创建输出目录
    json_dst.mkdir(parents=True, exist_ok=True)
    img_dst.mkdir(parents=True, exist_ok=True)

    # 收集所有 JSON 文件
    json_files = list(json_src.glob("**/*.json"))
    print(f"在 {json_src} 中找到 {len(json_files)} 个 JSON 文件")

    exported_count = 0
    for json_path in json_files:
        # 读取 JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        filtered_shapes = [s for s in shapes if s.get("label") in labels_set]

        if not filtered_shapes:
            continue

        # 更新 JSON
        data["shapes"] = filtered_shapes

        # 计算相对路径（保持子目录结构），如果没有子目录则直接使用文件名
        try:
            rel_path = json_path.relative_to(json_src)
        except ValueError:
            rel_path = json_path.name
        
        # 目标 JSON 路径
        dst_json = json_dst / rel_path
        dst_json.parent.mkdir(parents=True, exist_ok=True)

        # 保存 JSON
        with open(dst_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 查找对应的图片
        img_path = find_image_for_json(json_path, img_src)
        if img_path and img_path.exists():
            # 保持同样的相对目录结构（基于 JSON 的相对路径）
            # 但图片可能在不同目录结构，这里我们采用相对路径（如果 JSON 有子目录，图片也放在相应子目录）
            # 也可以选择仅使用文件名，但可能冲突。我们统一按 JSON 的相对路径存放图片
            dst_img = img_dst / rel_path.parent / img_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dst_img)
        else:
            print(f"警告: 未找到 JSON '{json_path}' 对应的图片，跳过图片复制")

        exported_count += 1

    print(f"导出完成！共导出 {exported_count} 个标注文件（及对应图片）")

def main():
    parser = argparse.ArgumentParser(description="按标签筛选导出 X-AnyLabeling 数据集，图片和 JSON 分开")
    parser.add_argument("--src-json", required=True, help="源 JSON 文件目录（可包含子目录）")
    parser.add_argument("--src-images", required=True, help="源图片文件目录")
    parser.add_argument("--dst-json", required=True, help="目标 JSON 输出目录")
    parser.add_argument("--dst-images", required=True, help="目标图片输出目录")
    parser.add_argument("--labels", required=True, help="要保留的标签，用逗号分隔，例如 person,car")
    parser.add_argument("--copy-others", action="store_true", help="将 src-json 目录下非 JSON 文件也复制到 dst-json（如 classes.txt）")

    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(',') if label.strip()]
    if not labels:
        print("错误: 必须指定至少一个标签")
        return

    export_dataset(args.src_json, args.src_images, args.dst_json, args.dst_images,
                   labels, copy_others=args.copy_others)

if __name__ == "__main__":
    main()