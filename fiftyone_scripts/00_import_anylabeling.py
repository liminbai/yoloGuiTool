import argparse

from anylabeling_import import import_images_with_anylabeling

DEFAULT_DATASET_NAME = "ppe_dataset"
DEFAULT_IMAGE_DIR = "/media/images/ppe"
DEFAULT_LABELS_DIR = "/media/images/ppe_xany"
DEFAULT_TAGS = ["raw_import"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="将图片与 X-AnyLabeling 标注（JSON）导入到 FiftyOne 数据集（初始导入入口）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "命令行示例:\n"
            f"  python 00_import_anylabeling.py --dataset-name {DEFAULT_DATASET_NAME} \\\n"
            f"      --image-dir {DEFAULT_IMAGE_DIR} \\\n"
            f"      --labels-dir {DEFAULT_LABELS_DIR} \\\n"
            "      --tags raw_import --overwrite\n"
            "\n"
            "说明:\n"
            "  - 若未指定 --image-dir / --labels-dir，脚本会以交互方式询问；\n"
            "  - 若未指定 --tags，默认使用 ['raw_import']。\n"
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=f"FiftyOne 数据集名称（默认: {DEFAULT_DATASET_NAME}）",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help=f"源图片所在目录，未指定时进入交互输入（默认: {DEFAULT_IMAGE_DIR}）",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help=f"X-AnyLabeling JSON 标注所在目录，未指定时进入交互输入（默认: {DEFAULT_LABELS_DIR}）",
    )
    parser.add_argument(
        "-t",
        "--tags",
        nargs="*",
        default=None,
        help=f"附加到导入样本的标签，可传多个，如 --tags raw_import incremental（默认: {DEFAULT_TAGS}）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若数据集已存在则先删除再全量重新导入（用于清空脏数据）",
    )
    return parser


def prompt_if_missing(args):
    """以交互方式补齐未通过命令行提供的目录与标签，便于直接运行。"""
    if not args.image_dir:
        value = input(f"请输入图片目录 [回车使用默认 {DEFAULT_IMAGE_DIR}]: ").strip()
        args.image_dir = value or DEFAULT_IMAGE_DIR

    if not args.labels_dir:
        value = input(f"请输入标注目录 [回车使用默认 {DEFAULT_LABELS_DIR}]: ").strip()
        args.labels_dir = value or DEFAULT_LABELS_DIR

    if not args.tags:
        value = input(f"请输入标签（多个用空格分隔）[回车使用默认 {' '.join(DEFAULT_TAGS)}]: ").strip()
        args.tags = value.split() if value else list(DEFAULT_TAGS)

    return args


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.image_dir and args.labels_dir):
        args = prompt_if_missing(args)

    tags = args.tags or list(DEFAULT_TAGS)
    print(f"📋 导入配置:\n"
          f"  dataset_name : {args.dataset_name}\n"
          f"  image_dir    : {args.image_dir}\n"
          f"  labels_dir   : {args.labels_dir}\n"
          f"  tags         : {tags}\n"
          f"  overwrite    : {args.overwrite}\n")

    import_images_with_anylabeling(
        dataset_name=args.dataset_name,
        image_dir=args.image_dir,
        labels_dir=args.labels_dir,
        tags=tags,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()