import json
from pathlib import Path

import fiftyone as fo

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def delete_dataset(dataset_name: str):
    """
    如果数据集存在，则将其从 FiftyOne 数据库中完全删除
    """
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        print(f"🗑️ 已删除数据集 [{dataset_name}]")
    else:
        print(f"⚠️ 数据集 [{dataset_name}] 不存在，跳过删除")


def ensure_ground_truth_field(dataset):
    schema = dataset.get_field_schema()
    if "ground_truth" not in schema:
        dataset.add_sample_field(
            "ground_truth",
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
            description="X-AnyLabeling detection labels",
        )
    return dataset


def create_or_load_dataset(dataset_name: str):
    if dataset_name in fo.list_datasets():
        dataset = fo.load_dataset(dataset_name)
        ensure_ground_truth_field(dataset)
        print(f"📂 已加载已有数据集 [{dataset_name}]，当前样本数: {len(dataset)}")
        return dataset

    dataset = fo.Dataset(dataset_name)
    ensure_ground_truth_field(dataset)
    print(f"✨ 已创建数据集 [{dataset_name}]")
    return dataset


def parse_anylabeling_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    image_width = float(annotations.get("imageWidth") or 1)
    image_height = float(annotations.get("imageHeight") or 1)
    detections = []

    for shape in annotations.get("shapes", []):
        label = (shape.get("label") or "").strip()
        points = shape.get("points") or []
        if not label or len(points) < 2:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

        width = max(x_max - x_min, 0.0)
        height = max(y_max - y_min, 0.0)
        if width <= 0 or height <= 0:
            continue

        # 转换为 FiftyOne 所需的归一化边界框 [xmin, ymin, width, height]
        detections.append(
            fo.Detection(
                label=label,
                bounding_box=[
                    x_min / image_width,
                    y_min / image_height,
                    width / image_width,
                    height / image_height,
                ],
            )
        )

    return fo.Detections(detections=detections)


def import_images_with_anylabeling(
    dataset_name: str, 
    image_dir: str, 
    labels_dir: str, 
    tags=None, 
    overwrite=False
):
    if tags is None:
        tags = ["raw_import"]
    elif isinstance(tags, str):
        tags = [tags]

    # 如果指定 overwrite=True，先全量删除已存在的脏数据集
    if overwrite:
        delete_dataset(dataset_name)

    dataset = create_or_load_dataset(dataset_name)
    ensure_ground_truth_field(dataset)

    image_root = Path(image_dir)
    labels_root = Path(labels_dir)

    existing_filepaths = {str(Path(sample.filepath).resolve()) for sample in dataset}
    new_count = 0

    # 1. 增量/批量添加图像
    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            continue

        image_abs_path = str(image_path.resolve())
        if image_abs_path in existing_filepaths:
            continue

        dataset.add_images([str(image_path)], tags=tags)
        existing_filepaths.add(image_abs_path)
        new_count += 1

    # 2. 绑定 X-AnyLabeling 标注信息
    updated_label_count = 0
    for sample in dataset:
        # 已存在有效的 ground_truth 则跳过
        if sample.ground_truth is not None and len(sample.ground_truth.detections) > 0:
            continue

        sample_path = Path(sample.filepath)
        json_path = labels_root / f"{sample_path.stem}.json"
        if not json_path.exists():
            continue

        parsed = parse_anylabeling_json(str(json_path))
        
        # 修正：判断 parsed.detections 列表的长度
        if len(parsed.detections) == 0:
            continue

        # 写入字段并显式持久化保存
        sample["ground_truth"] = parsed
        sample.save()  # 关键点：保存单个 sample 的修改
        updated_label_count += 1

    print(f"✅ 导入完成！新增图像: {new_count} 张，更新标注: {updated_label_count} 张，数据集当前总样本: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    import_images_with_anylabeling(
        dataset_name="ppe_dataset",
        image_dir="/media/images/ppe",
        labels_dir="/media/images/ppe_xany",
        tags=["raw_import"],
        overwrite=True,  # 首次测试如果想清空脏数据，建议设为 True
    )