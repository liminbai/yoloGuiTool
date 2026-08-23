import json
from pathlib import Path

import fiftyone as fo

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def delete_dataset(dataset_name: str):
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
        print(f"🗑️ 已删除数据集 [{dataset_name}]")
    else:
        print(f"⚠️ 数据集 [{dataset_name}] 不存在，跳过删除")


def parse_anylabeling_json(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    image_width = annotations.get("imageWidth") or 1
    image_height = annotations.get("imageHeight") or 1
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


def import_images_with_anylabeling(dataset_name: str, image_dir: str, labels_dir: str, tags=None):
    if tags is None:
        tags = ["raw_import"]
    elif isinstance(tags, str):
        tags = [tags]

    dataset = create_or_load_dataset(dataset_name)
    ensure_ground_truth_field(dataset)

    image_root = Path(image_dir)
    labels_root = Path(labels_dir)

    existing_filepaths = {str(Path(sample.filepath).resolve()) for sample in dataset}
    new_count = 0

    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            continue

        image_abs_path = str(image_path.resolve())
        if image_abs_path in existing_filepaths:
            continue

        dataset.add_images([str(image_path)], tags=tags)
        existing_filepaths.add(image_abs_path)
        new_count += 1

    for sample in dataset:
        try:
            existing_labels = sample["ground_truth"]
        except Exception:
            existing_labels = None

        if existing_labels is not None:
            continue

        sample_path = Path(sample.filepath)
        json_path = labels_root / f"{sample_path.stem}.json"
        if not json_path.exists():
            continue

        parsed = parse_anylabeling_json(str(json_path))
        if len(parsed) == 0:
            continue

        sample["ground_truth"] = parsed

    dataset.save()
    print(f"✅ 本次增量导入完成：新增图像 {new_count} 张，当前数据集样本数: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    import_images_with_anylabeling(
        dataset_name="ppe_dataset",
        image_dir="/media/images/ppe",
        labels_dir="/media/images/ppe_xany",
        tags=["raw_import"],
    )
