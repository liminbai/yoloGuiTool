#!/usr/bin/env python
import argparse
import os
import sys

import fiftyone as fo
import fiftyone.utils.random as four


DEFAULT_DATASET_NAME = "ppe_dataset"
DEFAULT_LABEL_FIELD = "ground_truth"
DEFAULT_CLASSES = ["person"]
DEFAULT_FORMAT = "yolo"
DEFAULT_EXPORT_DIR = "/exports"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.2


def build_parser():
    parser = argparse.ArgumentParser(
        description="将 FiftyOne 数据集以命令行方式导出为 YOLO 或 COCO/COCO-YAML 格式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python 02_export_yolo.py --dataset-name ppe_dataset --format yolo --classes helmet vest person --train-ratio 0.8 --val-ratio 0.2\n"
            "  python 02_export_yolo.py --dataset-name ppe_dataset --format coco-yaml --classes helmet vest person --export-dir /exports/coco/ppe_dataset\n"
        ),
    )

    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME,
                        help=f"FiftyOne 数据集名称（默认：{DEFAULT_DATASET_NAME}）")
    parser.add_argument("--label-field", type=str, default=DEFAULT_LABEL_FIELD,
                        help=f"数据集标注字段（默认：{DEFAULT_LABEL_FIELD}）")
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT,
                        choices=["yolo", "coco", "coco-yaml"],
                        help="导出的数据集格式：yolo / coco / coco-yaml（默认：yolo）")
    parser.add_argument("--export-dir", type=str, default=None,
                        help="导出目录；若不传则自动构建为 /exports/<format>/<dataset-name>")
    parser.add_argument("--classes", type=str, nargs="*", default=list(DEFAULT_CLASSES),
                        help="类别名列表，多个用空格分隔（默认：helmet vest person）")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help=f"训练集比例（默认：{DEFAULT_TRAIN_RATIO}）")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help=f"验证集比例（默认：{DEFAULT_VAL_RATIO}）")
    parser.add_argument("--split", type=str, nargs="*", default=["train", "val"],
                        help="导出时使用的保留拆分名单（默认：train val）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机划分种子（默认：42）")

    return parser


def resolve_format(format_name):
    format_map = {
        "yolo": (fo.types.YOLOv5Dataset, "yolo"),
        "coco": (fo.types.COCODetectionDataset, "coco"),
        "coco-yaml": (fo.types.COCODetectionDataset, "coco-yaml"),
    }

    if format_name not in format_map:
        raise ValueError(f"Unsupported format: {format_name}")

    dataset_type, export_key = format_map[format_name]
    return dataset_type, export_key


def validate_ratio(train_ratio, val_ratio):
    if train_ratio < 0 or val_ratio < 0:
        raise ValueError("train-ratio 和 val-ratio 必须为非负数")
    if abs((train_ratio + val_ratio) - 1.0) > 1e-6:
        raise ValueError("train-ratio + val-ratio 必须等于 1.0")


def default_export_dir(format_name, dataset_name):
    fmt_alias = "coco_yaml" if format_name == "coco-yaml" else format_name
    return os.path.join(DEFAULT_EXPORT_DIR, fmt_alias, dataset_name)


def write_coco_yaml(export_dir, classes):
    """写一个简化版的 COCO YAML 配置文件，方便在命令行导出后做格式说明。"""
    path = os.path.join(export_dir, "coco.yaml")
    os.makedirs(export_dir, exist_ok=True)

    lines = [
        "dataset_name: ppe_dataset",
        "format: coco",
        "train: train",
        "val: val",
        "num_classes: %d" % len(classes),
        "names:",
    ]

    for i, cls in enumerate(classes, start=0):
        lines.append(f"  {i}: {cls}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    validate_ratio(args.train_ratio, args.val_ratio)

    dataset = fo.load_dataset(args.dataset_name)
    if dataset is None:
        raise ValueError(f"FiftyOne 数据集不存在: {args.dataset_name}")

    valid_view = dataset.exists(args.label_field)
    if valid_view is None:
        raise ValueError(f"数据集 {args.dataset_name} 中没有标注字段 {args.label_field}")

    # 允许兼容一方工艺：若训练验证比例存在于 0/1 之外，采用指定参数拆分
    four.random_split(valid_view, {"train": args.train_ratio, "val": args.val_ratio}, seed=args.seed)

    dataset_type, export_key = resolve_format(args.format)

    export_dir = args.export_dir or default_export_dir(args.format, args.dataset_name)
    os.makedirs(export_dir, exist_ok=True)

    view = valid_view
    view.export(
        export_dir=export_dir,
        dataset_type=dataset_type,
        label_field=args.label_field,
        split=args.split,
        classes=args.classes,
    )

    if args.format in {"coco", "coco-yaml"}:
        # 生成一个独立的 YAML 元数据文件，供后续训练或配置引用
        yaml_path = write_coco_yaml(export_dir, args.classes)
        print(f"🚀 COCO 导出完成，YAML 元数据已写入: {yaml_path}")

    print(f"🚀 {args.format.upper()} 导出完成，目录: {export_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ 导出失败: {exc}", file=sys.stderr)
        raise
