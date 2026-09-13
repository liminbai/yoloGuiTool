#!/usr/bin/env python
"""将 FiftyOne 数据集按 train/val 随机拆分导出为 YOLO / COCO 格式，并生成 YAML 配置。

`yolo` 与 `coco-yaml` 都会导出成 **ultralytics 可直接训练** 的布局（txt 标注）::

    <export_dir>/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        dataset.yaml            # yolo 格式的配置文件
        coco.yaml               # coco-yaml 格式的配置文件（内容同样是 YOLO 训练配置）

    训练命令::

        yolo detect train data=<export_dir>/coco.yaml model=yolov8m.pt epochs=100 imgsz=640

为什么必须用这个布局（ultralytics 8.4 实测结论）：

1. ultralytics 只读取与图片同级的 ``labels/*.txt``，不读 COCO 的 ``labels.json``；
2. 它通过把图片路径中的 ``images`` 替换成 ``labels`` 来推导标注路径，
   所以图片必须放在 ``images/<split>``、标注必须放在 ``labels/<split>``；
3. 配置文件里不要写 ``path`` 键，ultralytics 会以 YAML 所在目录为基准解析
   ``train``/``val``，这样导出目录整体移动、或在容器/宿主机之间切换都不会失效。

`coco` 是纯 COCO 导出（json 标注），仅用于与其它框架交换数据，**不能**直接用于
``yolo detect train``::

    <export_dir>/
        train/
            data/*.jpg          # FiftyOne 默认把图片写入 data/
            labels.json         # COCO 标注文件
        val/
            data/*.jpg
            labels.json
        coco.yaml
"""
import argparse
import json
import os
import shutil
import sys

import fiftyone as fo
import fiftyone.utils.random as four


DEFAULT_DATASET_NAME = "ppe_dataset"
DEFAULT_LABEL_FIELD = "ground_truth"
# None 表示"自动从数据集中推断类别"。
# 注意：COCO 导出器会静默丢弃不在 classes 列表中的标签，
# 所以除非确实要裁剪类别，否则不要硬编码类别列表。
DEFAULT_CLASSES = None
DEFAULT_FORMAT = "yolo"
DEFAULT_EXPORT_DIR = "/exports"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SPLITS = ["train", "val"]

# COCO 标注文件名（FiftyOne 默认写在 export_dir 根目录下）
COCO_LABELS_FILENAME = "labels.json"
# 各格式图片所在子目录名，与 FiftyOne 各导出器的默认布局保持一致
YOLO_IMAGES_DIRNAME = "images"
COCO_IMAGES_DIRNAME = "data"
# 使用 images/<split> + labels/<split> 布局（ultralytics 可直接训练）的格式
YOLO_LAYOUT_FORMATS = {"yolo", "coco-yaml"}
# ultralytics 强制要求 train/val 同时存在，否则 check_det_dataset 直接报 SyntaxError
ULTRALYTICS_REQUIRED_SPLITS = ("train", "val")


def build_parser():
    parser = argparse.ArgumentParser(
        description="将 FiftyOne 数据集按 train/val 随机拆分导出为 YOLO 或 COCO/COCO-YAML 格式。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python 02_export_yolo.py --dataset-name ppe_dataset --format coco-yaml\n"
            "  python 02_export_yolo.py --dataset-name ppe_dataset --format coco-yaml --overwrite\n"
            "  python 02_export_yolo.py --dataset-name ppe_dataset --format coco-yaml "
            "--classes helmet vest person\n"
            "\n"
            "yolo / coco-yaml 导出目录结构（ultralytics 可直接训练）：\n"
            "  <export-dir>/images/train/*.jpg + images/val/*.jpg\n"
            "  <export-dir>/labels/train/*.txt + labels/val/*.txt\n"
            "  <export-dir>/coco.yaml    （或 dataset.yaml）\n"
            "\n"
            "  yolo detect train data=<export-dir>/coco.yaml model=yolov8m.pt epochs=100 imgsz=640\n"
            "\n"
            "coco 导出目录结构（仅供数据交换，不能直接训练）：\n"
            "  <export-dir>/train/data/*.jpg + labels.json\n"
            "  <export-dir>/val/data/*.jpg   + labels.json\n"
            "  <export-dir>/coco.yaml\n"
            "\n"
            "说明：\n"
            "  - 不传 --classes 时自动从数据集推断类别；\n"
            "  - COCO 导出器会丢弃不在 classes 中的标签，手写 --classes 时务必与数据一致。\n"
        ),
    )

    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME,
                        help=f"FiftyOne 数据集名称（默认：{DEFAULT_DATASET_NAME}）")
    parser.add_argument("--label-field", type=str, default=DEFAULT_LABEL_FIELD,
                        help=f"数据集标注字段（默认：{DEFAULT_LABEL_FIELD}）")
    parser.add_argument("--format", type=str, default=DEFAULT_FORMAT,
                        choices=["yolo", "coco", "coco-yaml"],
                        help="导出的数据集格式：yolo / coco / coco-yaml"
                             "（yolo 与 coco-yaml 都可用于 ultralytics 训练，默认：yolo）")
    parser.add_argument("--export-dir", type=str, default=None,
                        help=f"导出目录；若不传则自动构建为 {DEFAULT_EXPORT_DIR}/"
                             "<format>/<dataset-name>")
    parser.add_argument("--classes", type=str, nargs="*", default=DEFAULT_CLASSES,
                        help="类别名列表，多个用空格分隔；不传则自动从数据集推断")
    parser.add_argument("--splits", "--split", dest="splits", type=str, nargs="*",
                        default=list(DEFAULT_SPLITS),
                        help=f"拆分名称列表（默认：{' '.join(DEFAULT_SPLITS)}）；"
                             "yolo/coco-yaml 必须同时包含 train 和 val")
    parser.add_argument("--ratios", type=float, nargs="*", default=None,
                        help="各拆分比例，顺序与 --splits 一致；不传则使用 --train-ratio/--val-ratio")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                        help=f"训练集比例（默认：{DEFAULT_TRAIN_RATIO}）")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO,
                        help=f"验证集比例（默认：{DEFAULT_VAL_RATIO}）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机划分种子（默认：42）")
    parser.add_argument("--overwrite", action="store_true",
                        help="导出目录已存在时先清空再导出，避免残留上一次的图片/标注")
    parser.add_argument("--skip-missing-media", action="store_true",
                        help="跳过源图片缺失的样本而不是中止导出（这些样本的标注也会一并丢弃）")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="只导出随机抽取的 N 个样本，用于快速冒烟测试（默认：全部）")

    return parser


def resolve_format(format_name):
    """返回该格式对应的 FiftyOne 数据集类型。

    coco-yaml 也走 YOLOv5Dataset：ultralytics 只能训练 txt 标注，
    纯 COCO 的 labels.json 无法直接训练，因此这里不做 COCO 导出。
    """
    format_map = {
        "yolo": fo.types.YOLOv5Dataset,
        "coco": fo.types.COCODetectionDataset,
        "coco-yaml": fo.types.YOLOv5Dataset,
    }

    if format_name not in format_map:
        raise ValueError(f"Unsupported format: {format_name}")

    return format_map[format_name]


def resolve_ratios(args):
    """把命令行上的比例参数整理成与 --splits 等长的列表并归一化。"""
    ratios = list(args.ratios) if args.ratios else [args.train_ratio, args.val_ratio]

    if len(ratios) != len(args.splits):
        raise ValueError(
            f"--ratios 的个数({len(ratios)})必须与 --splits 的个数({len(args.splits)})一致"
        )

    if any(r < 0 for r in ratios):
        raise ValueError("拆分比例必须为非负数")

    total = sum(ratios)
    if total <= 0:
        raise ValueError("拆分比例之和必须大于 0")

    if abs(total - 1.0) > 1e-6:
        print(f"⚠️ 拆分比例之和为 {total}，已自动归一化", file=sys.stderr)

    return [r / total for r in ratios]


def uses_yolo_layout(format_name):
    """该格式是否导出为 images/<split> + labels/<split> 的 ultralytics 训练布局。"""
    return format_name in YOLO_LAYOUT_FORMATS


def yaml_filename(format_name):
    """返回该格式的配置文件名称（ultralytics 不关心文件名，只关心内容）。"""
    return "dataset.yaml" if format_name == "yolo" else "coco.yaml"


def validate_splits(format_name, splits):
    """ultralytics 要求 train 与 val 同时存在，提前拦截而不是等到训练时才报错。"""
    if not uses_yolo_layout(format_name):
        return

    missing = [s for s in ULTRALYTICS_REQUIRED_SPLITS if s not in splits]
    if missing:
        raise ValueError(
            f"{format_name} 导出用于 ultralytics 训练，--splits 必须同时包含 "
            f"{list(ULTRALYTICS_REQUIRED_SPLITS)}，当前缺少: {missing}"
        )


def default_export_dir(format_name, dataset_name):
    fmt_alias = "coco_yaml" if format_name == "coco-yaml" else format_name
    return os.path.join(DEFAULT_EXPORT_DIR, fmt_alias, dataset_name)


def clear_dir(path, overwrite):
    """YOLO 的各个 split 共用同一个 export_dir，导出前统一清理一次。

    注意：这里只删除、不预建目录，否则 FiftyOne 会误报
    "Directory ... already exists; export will be merged with existing files"。
    """
    if not os.path.isdir(path) or not os.listdir(path):
        return

    if overwrite:
        shutil.rmtree(path)
    else:
        print(
            f"⚠️ 目录已存在且非空: {path}，如需重新导出请加 --overwrite",
            file=sys.stderr,
        )


def resolve_classes(view, label_field, classes):
    """确定最终使用的类别列表。

    COCO 导出器对不在 classes 中的标签只告警并跳过，
    这里显式提示，避免出现"图片导出了、labels.json 里却没有标注"的情况。
    """
    observed = sorted(view.distinct(f"{label_field}.detections.label"))

    if not observed:
        raise ValueError(
            f"数据集 {view.dataset.name} 的 {label_field} 字段中没有 detections 标注"
        )

    if classes:
        missing = [c for c in observed if c not in classes]
        if missing:
            print(
                f"⚠️ 以下类别存在于数据集中但未包含在 --classes 中，导出时会被丢弃: {missing}",
                file=sys.stderr,
            )

        extra = [c for c in classes if c not in observed]
        if extra:
            print(f"ℹ️ --classes 中的以下类别在数据集中不存在: {extra}")

        return list(classes)

    print(f"ℹ️ 未指定 --classes，自动从数据集推断类别: {observed}")
    return observed


def check_media(view, skip_missing=False):
    """检查样本对应的源图片是否真实存在。

    FiftyOne 导出时会重新读取图片元数据并拷贝文件，只要有一张图缺失就会整体中断，
    因此这里提前检查：默认直接报错，加 --skip-missing-media 则剔除这些样本。
    """
    ids = view.values("id")
    paths = view.values("filepath")
    missing = [p for p in paths if not os.path.isfile(p)]

    if not missing:
        return view, []

    print(f"⚠️ 有 {len(missing)}/{len(paths)} 个样本的图片文件不存在:", file=sys.stderr)
    for path in missing[:20]:
        print(f"    {path}", file=sys.stderr)
    if len(missing) > 20:
        print(f"    ... 其余 {len(missing) - 20} 个略", file=sys.stderr)

    if not skip_missing:
        raise ValueError(
            "存在缺失的图片文件，导出会被 FiftyOne 中断。\n"
            "请先补齐这些图片，或加 --skip-missing-media 跳过这些样本"
            "（这些样本的标注也会一并丢弃）。"
        )

    keep_ids = [i for i, p in zip(ids, paths) if os.path.isfile(p)]
    print(f"ℹ️ 已跳过 {len(missing)} 个缺失媒体的样本，剩余 {len(keep_ids)} 个", file=sys.stderr)
    return view.select(keep_ids), missing


def split_view(view, splits, ratios, seed):
    """按比例随机拆分视图。

    注意：`four.random_split` 传入 dict 时只给样本打 tag 并返回 None，
    传入 list 时返回拆分后的视图元组；这里用 list 形式，避免污染数据集标签。
    """
    views = four.random_split(view, list(ratios), seed=seed)
    return dict(zip(splits, views))


def export_split(view, split, target_dir, format_name, dataset_type, label_field, classes,
                 overwrite=False, yolo_layout=None):
    """把单个拆分导出到 target_dir。"""
    if yolo_layout is None:
        yolo_layout = uses_yolo_layout(format_name)

    kwargs = dict(
        export_dir=target_dir,
        dataset_type=dataset_type,
        label_field=label_field,
    )

    if classes:
        kwargs["classes"] = classes

    if yolo_layout:
        # YOLOv5DatasetExporter 必须显式指定 split；COCO 导出器没有该参数（传了也只会被忽略）
        kwargs["split"] = split
        # YOLO 的多个 split 共用同一个 export_dir，overwrite 交给 prepare_dir 统一处理，
        # 否则导出第二个 split 时会把第一个 split 的成果删掉
    else:
        # COCO 每个 split 一个独立目录，交给 FiftyOne 原生 overwrite 逻辑清理
        kwargs["overwrite"] = overwrite

    view.export(**kwargs)
    return target_dir


def write_yaml_config(yaml_path, dataset_name, format_name, entries, classes):
    """写出 ultralytics 可直接训练的 YAML 配置。

    已用 ultralytics 8.4 实测验证过的三条约束：
    - 不写 ``path`` 键：ultralytics 会以 YAML 文件所在目录为基准解析 train/val，
      因此导出目录整体移动、或在容器/宿主机之间切换都不会失效；
    - ``train``/``val`` 指向 ``images/<split>``，ultralytics 会把路径中的 ``images``
      替换成 ``labels`` 再找同名 ``.txt``，所以标注必须落在 ``labels/<split>``；
    - 只有 ``train``/``val``/``nc``/``names`` 会被 ultralytics 读取，
      其余元信息一律写成注释，避免被误当作数据集键解析。

    Args:
        entries: [(split_name, relative_images_dir, ratio), ...]
    """
    lines = [
        "# 由 fiftyone_scripts/02_export_yolo.py 自动生成，可直接用于 ultralytics 训练：",
        "#   yolo detect train data=<该 yaml 的路径> model=yolov8m.pt epochs=100 imgsz=640",
        f"# dataset_name: {dataset_name}",
        f"# format: {format_name}",
        "",
    ]

    for split, rel_dir, ratio in entries:
        lines.append(f"# {split} 占比 {ratio:.4f}")
        lines.append(f"{split}: {rel_dir}")

    lines.append("")
    lines.append(f"nc: {len(classes)}")
    lines.append("names:")
    for name in classes:
        # 用 JSON 字符串做 YAML 双引号标量，避免类别名含 : # 等字符时解析出错
        lines.append(f"  - {json.dumps(str(name), ensure_ascii=False)}")

    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return yaml_path


def remove_stale_exporter_yaml(export_dir, keep_name):
    """删除导出器自己生成的 dataset.yaml。

    YOLOv5DatasetExporter 每导出一个 split 都会在 export_dir 根目录写一份
    dataset.yaml，后写的会覆盖先写的（只剩最后一个 split），内容也不含正确的
    train/val 结构。对 coco-yaml 来说它是一个会误导人的残留文件，直接清掉。
    """
    stale = os.path.join(export_dir, "dataset.yaml")
    if keep_name == "dataset.yaml" or not os.path.isfile(stale):
        return None

    os.remove(stale)
    return stale


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    format_name = args.format
    ratios = resolve_ratios(args)
    dataset_type = resolve_format(format_name)

    available = fo.list_datasets()
    if args.dataset_name not in available:
        raise ValueError(
            f"FiftyOne 数据集不存在: {args.dataset_name}（当前可用: {available}）"
        )

    dataset = fo.load_dataset(args.dataset_name)

    view = dataset.exists(args.label_field)
    if len(view) == 0:
        raise ValueError(f"数据集 {args.dataset_name} 中没有标注字段 {args.label_field}")

    num_objects = view.count(f"{args.label_field}.detections")
    if num_objects == 0:
        raise ValueError(
            f"数据集 {args.dataset_name} 的 {args.label_field} 字段中没有 detections 标注"
        )

    print(
        "📋 导出配置:\n"
        f"  dataset_name : {args.dataset_name}\n"
        f"  label_field  : {args.label_field}\n"
        f"  format       : {format_name}\n"
        f"  有效样本     : {len(view)}\n"
        f"  标注框       : {num_objects}"
    )

    classes = resolve_classes(view, args.label_field, args.classes)

    # 提前拦截源图片缺失的情况（FiftyOne 导出时遇到缺失图片会整体中断）
    view, _ = check_media(view, skip_missing=args.skip_missing_media)
    if len(view) == 0:
        raise ValueError("过滤掉缺失媒体的样本后，已经没有可导出的样本")

    if args.max_samples:
        if len(view) < args.max_samples:
            raise ValueError(
                f"--max-samples={args.max_samples} 大于可用样本数 {len(view)}"
            )
        view = view.take(args.max_samples, seed=args.seed)
        print(f"ℹ️ --max-samples 生效，仅导出随机 {len(view)} 个样本")

    export_dir = os.path.abspath(
        args.export_dir or default_export_dir(format_name, args.dataset_name)
    )

    yolo_layout = uses_yolo_layout(format_name)
    validate_splits(format_name, args.splits)

    # 随机拆分（返回视图，不修改数据集标签）
    split_views = split_view(view, args.splits, ratios, args.seed)
    ratio_by_split = dict(zip(args.splits, ratios))
    for split in args.splits:
        print(f"  {split}: {len(split_views[split])} 张")

    # yolo/coco-yaml 由导出器自行按 split 建子目录，两次导出都写到 export_dir 根目录，
    # 所以这里统一清理一次根目录；COCO 则每个 split 各自独立目录，
    # 由 FiftyOne 原生的 overwrite 逻辑负责清理。
    if yolo_layout:
        clear_dir(export_dir, args.overwrite)

    entries = []
    for split in args.splits:
        if yolo_layout:
            target_dir = export_dir
            rel_images_dir = os.path.join(YOLO_IMAGES_DIRNAME, split)
            overwrite = False
        else:
            target_dir = os.path.join(export_dir, split)
            rel_images_dir = os.path.join(split, COCO_IMAGES_DIRNAME)
            overwrite = args.overwrite

        export_split(
            split_views[split],
            split,
            target_dir,
            format_name,
            dataset_type,
            args.label_field,
            classes,
            overwrite=overwrite,
            yolo_layout=yolo_layout,
        )
        print(f"✅ {split} 导出完成: {target_dir}")

        entries.append((split, rel_images_dir, ratio_by_split[split]))

    yaml_name = yaml_filename(format_name)
    yaml_path = write_yaml_config(
        os.path.join(export_dir, yaml_name),
        args.dataset_name,
        format_name,
        entries,
        classes,
    )

    if yolo_layout:
        stale = remove_stale_exporter_yaml(export_dir, yaml_name)
        if stale:
            print(f"ℹ️ 已移除导出器生成的残留配置: {stale}")

        print("\n📂 导出的训练目录结构:")
        print(f"  {export_dir}")
        print(f"  ├── images/{', images/'.join(args.splits)}")
        print(f"  ├── labels/{', labels/'.join(args.splits)}")
        print(f"  └── {yaml_name}")
        print("\n🚀 训练命令:")
        print(f"  yolo detect train data={yaml_path} model=yolov8m.pt "
              "epochs=100 imgsz=640 batch=16 device=0")
    else:
        labels_paths = ", ".join(
            os.path.join(split, COCO_LABELS_FILENAME) for split in args.splits
        )
        print(f"ℹ️ COCO 标注文件: {labels_paths}")
        print("ℹ️ coco 为纯 COCO 格式，无法直接用于 yolo detect train；"
              "需要训练请改用 --format coco-yaml")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ 导出失败: {exc}", file=sys.stderr)
        raise
