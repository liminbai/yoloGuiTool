"""类别文件加载工具。

统一处理 GUI 中“类别 -> 从文件加载”的解析逻辑，重点支持
``fiftyone_scripts/02_export_yolo.py`` 导出的 coco.yaml（Ultralytics 数据配置）::

    # dataset_name: ppe_dataset
    # format: coco-yaml
    train: images/train
    val: images/val
    nc: 11
    names:
      - "boots"
      - "gloves"
      ...

同时兼容以下常见格式：

* YOLO 数据 yaml/json：``names`` 为列表，或 ``{0: 'person', 1: 'car'}`` 索引字典；
* COCO 标注 json：``categories: [{"id": 0, "name": "person"}, ...]``；
* 纯列表 json/yaml：``["person", "car"]``；
* 文本文件：每行一个类别，支持 ``0: person`` / ``0 person`` 序号前缀与 ``#`` 注释。
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import yaml

#: 项目根目录（yoloGuiTool/）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: coco.yaml 等导出文件所在目录，作为“从文件加载”的默认打开位置
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "exports", "coco_yaml")

#: 类别文件选择对话框的过滤器
CLASS_FILE_FILTER = (
    "类别文件 (*.yaml *.yml *.json *.txt);;"
    "COCO/YOLO 配置 (*.yaml *.yml);;"
    "COCO 标注 (*.json);;"
    "文本文件 (*.txt);;"
    "所有文件 (*)"
)

#: 支持带序号的类别行，如 "0: person" / "0. person" / "0、person" / "0-person"
_INDEX_PREFIX_RE = re.compile(r"^\s*\d+\s*[:.、)\-]\s*(.+)$")
#: 支持空白分隔的序号，如 "0 person"
_NUM_SPACE_RE = re.compile(r"^\s*\d+\s+(\S.*)$")


def default_dir() -> str:
    """返回文件选择对话框的默认目录（不存在时返回空串）。"""
    return DEFAULT_DIR if os.path.isdir(DEFAULT_DIR) else ""


def load_classes(file_path: str) -> Tuple[List[str], List[str]]:
    """从文件读取类别名列表。

    Args:
        file_path: 类别文件路径（``.yaml``/``.yml``/``.json``/``.txt``）。

    Returns:
        ``(classes, warnings)``：类别名列表（顺序即索引顺序）与警告信息列表。

    Raises:
        ValueError: 文件为空、解析失败或未解析到任何类别。
    """
    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    if not content:
        raise ValueError("文件内容为空")

    suffix = os.path.splitext(file_path)[1].lower()

    if suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML 解析失败: {e}") from e
        classes, warnings = _extract_classes(data)
    elif suffix == ".json":
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 解析失败: {e}") from e
        classes, warnings = _extract_classes(data)
    else:
        classes, warnings = _parse_text(content), []

    classes = [str(c).strip() for c in classes if str(c).strip()]
    if not classes:
        raise ValueError("未从文件中解析到任何类别")

    return classes, warnings


def _extract_classes(data: Any) -> Tuple[List[str], List[str]]:
    """从已解析的 yaml/json 数据结构中提取类别名。"""
    # 纯列表：["person", "car"] 或 COCO categories: [{"id": 0, "name": "person"}]
    if isinstance(data, list):
        if data and all(isinstance(item, dict) for item in data):
            if not all("name" in item for item in data):
                raise ValueError("列表元素不是 COCO categories 结构（缺少 name 字段）")
            return _categories_to_names(data), []
        return [str(item) for item in data], []

    if isinstance(data, dict):
        # COCO 标注 json
        if "categories" in data:
            return _categories_to_names(data["categories"]), []

        # coco.yaml / YOLO data yaml：核心字段 names
        if "names" in data:
            names = data["names"]
            if isinstance(names, dict):
                classes = _mapping_to_names(names)
            elif isinstance(names, list):
                classes = [str(item) for item in names]
            else:
                raise ValueError(f"names 字段类型不支持: {type(names).__name__}")
            return classes, _check_nc(data, classes)

        # 仅含索引映射：{0: 'person', 1: 'car'}
        if data and all(isinstance(k, int) or str(k).isdigit() for k in data):
            return _mapping_to_names(data), []

        keys = ", ".join(str(k) for k in data.keys()) if data else "（空映射）"
        raise ValueError(f"未找到 names / categories 字段（文件中的字段: {keys}）")

    raise ValueError(f"不支持的数据结构: {type(data).__name__}")


def _categories_to_names(categories: Any) -> List[str]:
    """把 COCO ``categories`` 转为按 id 排序的类别名列表。"""
    items = [c for c in categories if isinstance(c, dict) and "name" in c]
    if not items:
        raise ValueError("categories 中没有有效的 name 字段")
    if all(str(c.get("id", "")).lstrip("-").isdigit() for c in items):
        items = sorted(items, key=lambda c: int(c["id"]))
    return [str(c["name"]) for c in items]


def _mapping_to_names(mapping: Dict[Any, Any]) -> List[str]:
    """把 ``{索引: 类别名}`` 映射按索引排序为类别名列表。"""
    numeric: List[Tuple[int, Any]] = []
    others: List[Tuple[str, Any]] = []
    for key, value in mapping.items():
        key_str = str(key)
        if key_str.isdigit():
            numeric.append((int(key_str), value))
        else:
            others.append((key_str, value))

    numeric.sort(key=lambda item: item[0])
    others.sort(key=lambda item: item[0])
    return [str(value) for _, value in numeric + others]


def _check_nc(data: Dict[Any, Any], classes: List[str]) -> List[str]:
    """校验 coco.yaml 中的 ``nc`` 与 ``names`` 数量是否一致。"""
    nc = data.get("nc")
    if isinstance(nc, str) and nc.isdigit():
        nc = int(nc)
    if isinstance(nc, int) and nc != len(classes):
        return [f"文件声明 nc={nc}，但解析出 {len(classes)} 个类别，已按 names 内容加载。"]
    return []


def _parse_text(content: str) -> List[str]:
    """解析纯文本类别文件（每行一个类别，可带序号前缀）。"""
    classes: List[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        for pattern in (_INDEX_PREFIX_RE, _NUM_SPACE_RE):
            matched = pattern.match(line)
            if matched:
                line = matched.group(1).strip()
                break
        line = line.strip().strip(",").strip().strip('"').strip("'").strip()
        if line:
            classes.append(line)
    return classes
