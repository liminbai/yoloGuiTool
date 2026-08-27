#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 格式标注批量转换为 X-AnyLabeling (LabelMe) JSON 格式
支持从 YAML 文件读取类别名称，支持递归处理子目录（可选）
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import cv2
import yaml

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== YAML 类别加载 ====================
def load_classes_from_yaml(yaml_path: Union[str, Path]) -> List[str]:
    """
    从 COCO/YOLOv8 风格的 YAML 文件中加载类别名称列表。

    支持的格式：
        - names: {0: person, 1: bicycle, ...}
        - names: [person, bicycle, ...]
        - names: "person\\nbicycle\\ncar"   (多行字符串)

    Args:
        yaml_path: YAML 文件路径

    Returns:
        类别名称列表（按索引顺序）

    Raises:
        ValueError: 如果 YAML 中没有 'names' 或 'classes' 字段，或格式不支持
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    raw_names = data.get('names', data.get('classes'))
    if raw_names is None:
        raise ValueError(f"YAML 文件中未找到 'names' 或 'classes' 字段: {yaml_path}")

    if isinstance(raw_names, dict):
        # 字典格式：{0: 'person', 1: 'bicycle', ...}，按整数键排序
        ordered_names = []
        for key in sorted(raw_names.keys(), key=lambda k: int(k)):
            value = raw_names[key]
            if isinstance(value, str):
                ordered_names.append(value)
            else:
                raise ValueError(f"YAML 'names' 中键 {key} 对应的值不是字符串: {value!r}")
        return ordered_names

    if isinstance(raw_names, list):
        return [str(item) for item in raw_names]

    if isinstance(raw_names, str):
        # 按行分割，过滤空行
        return [line.strip() for line in raw_names.splitlines() if line.strip()]

    raise TypeError(f"不支持的类别格式: {type(raw_names).__name__}")


# ==================== YOLO 行解析与转换 ====================
def parse_yolo_line(line: str) -> Optional[Dict[str, float]]:
    """
    解析 YOLO 格式的一行标注。

    YOLO 格式: class_id x_center y_center width height (全部归一化)

    Args:
        line: 一行文本

    Returns:
        字典包含 class_id, x_center, y_center, w, h
        如果解析失败则返回 None
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError as e:
        logger.warning(f"解析 YOLO 行失败: {line.strip()} (错误: {e})")
        return None

    # 检查归一化坐标是否在 [0,1] 范围内
    if not all(0.0 <= v <= 1.0 for v in (x_center, y_center, w, h)):
        logger.warning(f"归一化坐标越界: {line.strip()}")
        return None

    return {
        'class_id': class_id,
        'x_center': x_center,
        'y_center': y_center,
        'w': w,
        'h': h
    }


def yolo_to_labelme_shape(data: Dict[str, float],
                          class_name: str,
                          img_width: int,
                          img_height: int) -> Dict[str, Any]:
    """
    将 YOLO 归一化数据转换为 LabelMe shape 对象。

    Args:
        data: 包含 x_center, y_center, w, h 的字典
        class_name: 类别名称字符串
        img_width: 图像宽度（像素）
        img_height: 图像高度（像素）

    Returns:
        LabelMe shape 字典
    """
    # 计算绝对坐标（像素），并裁剪到图像边界
    abs_w = data['w'] * img_width
    abs_h = data['h'] * img_height
    abs_cx = data['x_center'] * img_width
    abs_cy = data['y_center'] * img_height

    xmin = max(0.0, abs_cx - abs_w / 2.0)
    ymin = max(0.0, abs_cy - abs_h / 2.0)
    xmax = min(float(img_width), abs_cx + abs_w / 2.0)
    ymax = min(float(img_height), abs_cy + abs_h / 2.0)

    # 如果裁剪后宽度或高度为 0，抛出异常或返回 None（这里选择返回 None 由调用方处理）
    if xmax <= xmin or ymax <= ymin:
        return None

    return {
        "label": class_name,
        "score": None,
        "points": [[xmin, ymin], [xmax, ymax]],
        "group_id": None,
        "description": "",
        "shape_type": "rectangle",
        "flags": {},
        "direction": 0.0
    }


# ==================== 批量转换主函数 ====================
def yolo_to_x_anylabeling(
    img_dir: Union[str, Path],
    txt_dir: Union[str, Path],
    json_out_dir: Union[str, Path],
    classes: List[str],
    recursive: bool = False,
) -> int:
    """
    将 YOLO txt 标注批量转换为 X-AnyLabeling JSON 格式。

    Args:
        img_dir: 图片目录
        txt_dir: YOLO txt 标注目录
        json_out_dir: 输出 JSON 目录
        classes: 类别名称列表（索引对应 class_id）
        recursive: 是否递归处理子目录（图片和 txt 子目录结构需一致）

    Returns:
        成功转换的 JSON 文件数量
    """
    img_dir = Path(img_dir)
    txt_dir = Path(txt_dir)
    json_out_dir = Path(json_out_dir)
    json_out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists():
        logger.error(f"图片目录不存在: {img_dir}")
        return 0
    if not txt_dir.exists():
        logger.error(f"TXT 标注目录不存在: {txt_dir}")
        return 0

    # 收集所有图片文件（支持常见格式）
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    if recursive:
        img_files = [p for p in img_dir.rglob('*') if p.suffix.lower() in valid_exts]
    else:
        img_files = [p for p in img_dir.glob('*') if p.suffix.lower() in valid_exts]

    logger.info(f"找到 {len(img_files)} 张图片")

    success_count = 0
    for img_path in img_files:
        # 获取相对于 img_dir 的相对路径，用于保持目录结构
        try:
            rel_path = img_path.relative_to(img_dir)
        except ValueError:
            rel_path = img_path.name

        # 对应的 txt 文件路径（保持相同相对路径，扩展名替换为 .txt）
        txt_path = txt_dir / rel_path.with_suffix('.txt')

        # 读取图片尺寸（用灰度图读取，速度快）
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"无法读取图片: {img_path}，跳过")
            continue
        img_height, img_width = img.shape[:2]

        # 构建 LabelMe JSON 数据结构
        labelme_data = {
            "version": "0.4.0",
            "flags": {},
            "shapes": [],
            "imagePath": img_path.name,  # 仅文件名，不含路径
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width
        }

        # 如果对应的 txt 文件存在，解析并添加 shapes
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                parsed = parse_yolo_line(line)
                if parsed is None:
                    continue

                class_id = parsed['class_id']
                # 获取类别名称
                if 0 <= class_id < len(classes):
                    class_name = classes[class_id]
                else:
                    logger.warning(f"类别 ID {class_id} 超出范围，使用数字标签: {img_path}")
                    class_name = str(class_id)

                shape = yolo_to_labelme_shape(parsed, class_name, img_width, img_height)
                if shape is not None:
                    labelme_data["shapes"].append(shape)

        # 写入 JSON 文件（保持相同的相对路径结构）
        json_path = json_out_dir / rel_path.with_suffix('.json')
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        success_count += 1

    logger.info(f"处理完成！成功生成 {success_count} 个 JSON 文件")
    return success_count


# ==================== 命令行入口 ====================
def main():
    parser = argparse.ArgumentParser(
        description="YOLO 标注批量转换为 X-AnyLabeling (LabelMe) JSON 格式"
    )
    parser.add_argument(
        "--img-dir", required=True,
        help="图片目录路径"
    )
    parser.add_argument(
        "--txt-dir", required=True,
        help="YOLO txt 标注目录路径"
    )
    parser.add_argument(
        "--json-out", required=True,
        help="输出 JSON 目录路径"
    )
    parser.add_argument(
        "--classes-yaml", required=True,
        help="包含类别名称的 YAML 文件路径"
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="是否递归处理子目录（图片和标注子目录结构需一致）"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )

    args = parser.parse_args()

    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))

    try:
        classes = load_classes_from_yaml(args.classes_yaml)
        logger.info(f"加载了 {len(classes)} 个类别: {classes}")
    except Exception as e:
        logger.error(f"加载类别 YAML 失败: {e}")
        return 1

    count = yolo_to_x_anylabeling(
        img_dir=args.img_dir,
        txt_dir=args.txt_dir,
        json_out_dir=args.json_out,
        classes=classes,
        recursive=args.recursive,
    )

    if count == 0:
        logger.warning("没有生成任何 JSON 文件，请检查输入数据。")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())