import json
from pathlib import Path

import fiftyone as fo

ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Sample 字段名
DETECTION_FIELD = "ground_truth"
POLYGON_FIELD = "ground_truth_polygons"

# 只写矩形框（Detection）的 shape_type
_BOX_SHAPE_TYPES = {"rectangle", "rect", "box"}
# 写矩形框 + 多边形轮廓（Detection + Polyline）的 shape_type
# rotation 为四点旋转框，轴对齐 bbox 会丢角度，因此额外保留其多边形轮廓
_POLYGON_SHAPE_TYPES = {"polygon", "rotation"}
# 只写开放曲线（Polyline，不闭合、不填充）的 shape_type
_LINE_SHAPE_TYPES = {"line", "polyline", "linestrip"}
_SUPPORTED_SHAPE_TYPES = _BOX_SHAPE_TYPES | _POLYGON_SHAPE_TYPES | _LINE_SHAPE_TYPES


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
    """确保标注字段存在：ground_truth(Detections) + ground_truth_polygons(Polylines)。"""
    schema = dataset.get_field_schema()
    if DETECTION_FIELD not in schema:
        dataset.add_sample_field(
            DETECTION_FIELD,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Detections,
            description="X-AnyLabeling detection labels (bounding boxes)",
        )
    if POLYGON_FIELD not in schema:
        dataset.add_sample_field(
            POLYGON_FIELD,
            fo.EmbeddedDocumentField,
            embedded_doc_type=fo.Polylines,
            description="X-AnyLabeling polygon/polyline labels",
        )
    return dataset


# 兼容旧调用方（dataLoad.py 等）
ensure_label_fields = ensure_ground_truth_field


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


def resolve_image_size(annotations, image_path=None):
    """优先使用 JSON 中的 imageWidth/imageHeight，缺失时回退到直接读取图片头。"""
    try:
        width = float(annotations.get("imageWidth") or 0)
        height = float(annotations.get("imageHeight") or 0)
    except (TypeError, ValueError):
        width = height = 0.0

    if width > 0 and height > 0:
        return width, height

    if image_path:
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                return float(img.width), float(img.height)
        except Exception:
            pass

    return 0.0, 0.0


def _extract_xy(points):
    """提取二维像素坐标，自动忽略维度异常的脏点（如 3 元组）。"""
    coords = []
    for point in points or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            coords.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return coords


def _normalize_points(points, width, height):
    """像素坐标 -> 归一化坐标：裁剪到 [0,1]，并去除相邻重复点与收尾闭合重复点。"""
    normalized = []
    for x, y in points:
        nx = min(max(x / width, 0.0), 1.0)
        ny = min(max(y / height, 0.0), 1.0)
        if normalized and normalized[-1] == [nx, ny]:
            continue
        normalized.append([nx, ny])

    while len(normalized) > 1 and normalized[0] == normalized[-1]:
        normalized.pop()

    return normalized


def _bbox_from_points(points, width, height):
    """像素坐标 -> FiftyOne 归一化 [xmin, ymin, width, height]；退化框返回 None。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    box_width = x_max - x_min
    box_height = y_max - y_min
    if box_width <= 0 or box_height <= 0:
        return None

    return [
        min(max(x_min / width, 0.0), 1.0),
        min(max(y_min / height, 0.0), 1.0),
        min(box_width / width, 1.0),
        min(box_height / height, 1.0),
    ]


def parse_anylabeling_annotations(json_path, image_path=None):
    """解析 X-AnyLabeling JSON，返回 (fo.Detections, fo.Polylines, stats)。

    - rectangle / rect / box  -> Detection（轴对齐 bbox）
    - polygon / rotation      -> Detection + Polyline（closed=True, filled=True）
    - line / polyline         -> Polyline（开放曲线）
    - 其它 shape_type         -> 退化为 Detection，并在 stats["unsupported"] 中计数
    """
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    stats = {
        "detections": 0,
        "polylines": 0,
        "skipped": 0,
        "unsupported": {},
        "missing_size": False,
    }

    width, height = resolve_image_size(annotations, image_path)
    if width <= 0 or height <= 0:
        stats["missing_size"] = True
        return fo.Detections(detections=[]), fo.Polylines(polylines=[]), stats

    detections = []
    polylines = []

    for shape in annotations.get("shapes") or []:
        label = (shape.get("label") or "").strip()
        shape_type = str(shape.get("shape_type") or "polygon").lower()
        points = _extract_xy(shape.get("points"))

        if not label or len(points) < 2:
            stats["skipped"] += 1
            continue

        if shape_type not in _SUPPORTED_SHAPE_TYPES:
            stats["unsupported"][shape_type] = stats["unsupported"].get(shape_type, 0) + 1

        bounding_box = _bbox_from_points(points, width, height)
        if bounding_box is not None:
            detections.append(fo.Detection(label=label, bounding_box=bounding_box))
            stats["detections"] += 1

        if shape_type in _POLYGON_SHAPE_TYPES:
            normalized = _normalize_points(points, width, height)
            if len(normalized) < 3:
                continue
            polylines.append(
                fo.Polyline(
                    # 注意：FiftyOne >= 1.x 中 points 为「多段轮廓」列表，需再嵌套一层
                    points=[normalized],
                    label=label,
                    closed=True,
                    filled=True,
                )
            )
            stats["polylines"] += 1
        elif shape_type in _LINE_SHAPE_TYPES:
            normalized = _normalize_points(points, width, height)
            if len(normalized) < 2:
                continue
            polylines.append(
                fo.Polyline(
                    points=[normalized],
                    label=label,
                    closed=False,
                    filled=False,
                )
            )
            stats["polylines"] += 1

    return (
        fo.Detections(detections=detections),
        fo.Polylines(polylines=polylines),
        stats,
    )


def parse_anylabeling_json(json_path, image_path=None):
    """兼容旧接口：仅返回 Detections。"""
    detections, _polylines, _stats = parse_anylabeling_annotations(json_path, image_path)
    return detections


def _has_labels(sample):
    detections = sample.get_field(DETECTION_FIELD)
    polygons = sample.get_field(POLYGON_FIELD)
    return bool(
        (detections is not None and len(detections.detections) > 0)
        or (polygons is not None and len(polygons.polylines) > 0)
    )


def import_images_with_anylabeling(
    dataset_name: str,
    image_dir: str,
    labels_dir: str,
    tags=None,
    overwrite=False,
    refresh_labels=False,
):
    """导入/增量更新图片与 X-AnyLabeling 标注。

    Args:
        refresh_labels: 忽略已有标注，重新解析所有样本 JSON 并覆盖写入。
            用于给历史数据集补齐 ground_truth_polygons 字段或同步标注修改。
    """
    if tags is None:
        tags = ["raw_import"]
    elif isinstance(tags, str):
        tags = [tags]

    # 如果指定 overwrite=True，先全量删除已存在的脏数据集
    if overwrite:
        delete_dataset(dataset_name)

    dataset = create_or_load_dataset(dataset_name)

    image_root = Path(image_dir)
    labels_root = Path(labels_dir)

    existing_filepaths = {str(Path(sample.filepath).resolve()) for sample in dataset}

    # 1. 批量新增图像（一次性写入，避免逐张 add_images 的开销）
    new_image_paths = []
    for image_path in sorted(image_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            continue

        image_abs_path = str(image_path.resolve())
        if image_abs_path in existing_filepaths:
            continue

        existing_filepaths.add(image_abs_path)
        new_image_paths.append(str(image_path))

    if new_image_paths:
        dataset.add_images(new_image_paths, tags=tags)

    # 2. 解析标注并批量写回
    detections_map = {}
    polygons_map = {}
    updated_count = 0
    polygon_count = 0
    missing_json = 0
    empty_json = 0
    failed = 0
    unsupported_total = {}

    for sample in dataset:
        if not refresh_labels and _has_labels(sample):
            continue

        sample_path = Path(sample.filepath)
        json_path = labels_root / f"{sample_path.stem}.json"
        if not json_path.exists():
            missing_json += 1
            continue

        try:
            detections, polygons, stats = parse_anylabeling_annotations(
                str(json_path), sample.filepath
            )
        except (OSError, ValueError) as exc:
            failed += 1
            print(f"⚠️ 解析失败 {json_path.name}: {exc}")
            continue

        if stats["missing_size"]:
            failed += 1
            print(f"⚠️ 缺少图像尺寸信息，跳过 {json_path.name}")
            continue

        for shape_type, count in stats["unsupported"].items():
            unsupported_total[shape_type] = unsupported_total.get(shape_type, 0) + count

        if refresh_labels:
            # 刷新模式：即使标注为空也写入，以便清除历史脏标注
            detections_map[sample.id] = detections if detections.detections else None
            polygons_map[sample.id] = polygons if polygons.polylines else None
        else:
            if not detections.detections and not polygons.polylines:
                empty_json += 1
                continue
            if detections.detections:
                detections_map[sample.id] = detections
            if polygons.polylines:
                polygons_map[sample.id] = polygons

        updated_count += 1
        polygon_count += stats["polylines"]

    if detections_map:
        dataset.set_values(DETECTION_FIELD, detections_map, key_field="id")
    if polygons_map:
        dataset.set_values(POLYGON_FIELD, polygons_map, key_field="id")

    print(
        f"✅ 导入完成！新增图像: {len(new_image_paths)} 张，"
        f"更新标注: {updated_count} 张（其中多边形 {polygon_count} 个），"
        f"数据集当前总样本: {len(dataset)}"
    )

    if missing_json or empty_json or failed or unsupported_total:
        detail = []
        if missing_json:
            detail.append(f"无对应 JSON: {missing_json}")
        if empty_json:
            detail.append(f"JSON 内无有效标注: {empty_json}")
        if failed:
            detail.append(f"解析失败/缺少尺寸: {failed}")
        if unsupported_total:
            detail.append(f"未识别的 shape_type（按矩形框处理）: {unsupported_total}")
        print("ℹ️ 跳过明细 -> " + "，".join(detail))

    return dataset


if __name__ == "__main__":
    import_images_with_anylabeling(
        dataset_name="ppe_dataset",
        image_dir="/media/images/ppe",
        labels_dir="/media/images/ppe_xany",
        tags=["raw_import"],
        overwrite=True,  # 首次测试如果想清空脏数据，建议设为 True
    )
