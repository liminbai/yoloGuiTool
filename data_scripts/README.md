# data_scripts 数据标注脚本

本目录存放用于 **X-AnyLabeling / LabelMe 格式标注数据** 的批量处理脚本，
涵盖常见标注格式转换（YOLO / VOC）与数据集整理（按标签导出、拷贝、重命名、字段修正、目录同步）等场景。

> 说明：本目录中的脚本统一以 `*.json` 为 X-AnyLabeling / LabelMe 标注文件
> （包含 `imagePath`、`imageWidth`、`imageHeight`、`shapes` 等字段）。

## 脚本一览

| 脚本 | 功能 | 是否需要第三方依赖 |
| --- | --- | --- |
| [yolo_to_x_anylabeling.py](#1-yolo_to_x_anylabelingpy) | YOLO txt → X-AnyLabeling JSON | 需要 `opencv-python`、`pyyaml` |
| [voc_to_x_anylabeling.py](#2-voc_to_x_anylabelingpy) | Pascal VOC XML → X-AnyLabeling JSON | 仅标准库 |
| [export_by_labels.py](#3-export_by_labelspy) | 按标签筛选导出数据集 | 仅标准库 |
| [copy_json_by_images.py](#4-copy_json_by_imagespy) | 按图片批量拷贝同名 JSON | 仅标准库 |
| [rename_xanylabeling_separate.py](#5-rename_xanylabeling_separatepy) | 批量重命名图片与 JSON（分离目录） | 仅标准库 |
| [fix_label.py](#6-fix_labelpy) | 修正 imagePath / 递归替换字段 | 仅标准库 |
| [sync_left_to_right.py](#7-sync_left_to_rightpy) | 以左目录为基准清理右目录多余文件 | 仅标准库 |

> **提醒**：多数会改动文件系统的脚本都提供了 `--dry-run`（预览）模式，
> 建议正式执行前先预览一遍结果。

---

## 1. yolo_to_x_anylabeling.py

将 YOLO txt 标注批量转换为 X-AnyLabeling (LabelMe) JSON 格式，需读取图片以获得实际尺寸。

**依赖**：

```bash
pip install opencv-python pyyaml
```

**用法**：

```bash
python yolo_to_x_anylabeling.py \
    --img-dir /path/to/images \
    --txt-dir /path/to/labels \
    --json-out /path/to/output \
    --classes-yaml /path/to/classes.yaml
```

**参数**：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `--img-dir` | ✅ | 图片目录 |
| `--txt-dir` | ✅ | YOLO txt 标注目录 |
| `--json-out` | ✅ | 输出 JSON 目录（自动创建） |
| `--classes-yaml` | ✅ | 类别名称 YAML 文件（支持 `names`/`classes` 字段） |
| `--recursive` | - | 递归处理子目录（图片与 txt 目录结构需一致） |
| `--log-level` | - | 日志级别：`DEBUG/INFO/WARNING/ERROR`，默认 `INFO` |

**补充说明**：

- `--classes-yaml` 的 `names` 支持字典 `{0: person, ...}`、列表 `[person, ...]`、多行字符串三种写法。
- YOLO 坐标为归一化格式，转换时会换算成像素并裁剪到图片边界内。
- 输出 JSON 会保持与图片相同的相对目录结构；`imagePath` 仅写入图片文件名。
- 图片扩展名支持 `.jpg/.jpeg/.png/.bmp`（不区分大小写）。

---

## 2. voc_to_x_anylabeling.py

将 Pascal VOC XML 标注批量转换为 X-AnyLabeling (LabelMe) JSON 格式。

**用法**：

```bash
python voc_to_x_anylabeling.py <xml_dir> <json_output_dir>
```

**参数**（位置参数）：

| 参数 | 说明 |
| --- | --- |
| `xml_dir` | 存放 VOC XML 文件的输入目录 |
| `json_output_dir` | 输出 JSON 目录（不存在会自动创建） |

**补充说明**：

- 只读取当前目录下的 `.xml` 文件（不递归）。
- 生成的每个目标框为 `shape_type: "rectangle"` 的 LabelMe shape。
- 跳过缺少 `<filename>`、`<size>` 或边界框解析失败的文件。

---

## 3. export_by_labels.py

按标签筛选导出 X-AnyLabeling 数据集，图片与 JSON 分目录存放，
导出后仅保留包含指定标签的 JSON 及其对应图片（过滤掉其他标签的目标）。

**用法**：

```bash
python export_by_labels.py \
    --src-json /path/to/jsons \
    --src-images /path/to/images \
    --dst-json /path/to/output/jsons \
    --dst-images /path/to/output/images \
    --labels person,car,bus
```

**参数**：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `--src-json` | ✅ | 源 JSON 目录（可含子目录） |
| `--src-images` | ✅ | 源图片目录 |
| `--dst-json` | ✅ | 目标 JSON 输出目录 |
| `--dst-images` | ✅ | 目标图片输出目录 |
| `--labels` | ✅ | 保留的标签，逗号分隔，如 `person,car` |
| `--copy-others` | - | 同时把源 JSON 目录中的非 JSON 文件（如 `classes.txt`）复制到目标 |

**补充说明**：

- 仅包含目标标签中**任意一个**的 JSON 才会被导出；导出时 `shapes` 只保留命中标签的目标。
- JSON 的相对子目录结构会被保留；对应图片按 JSON 的相对路径存放。
- 图片定位优先依据 JSON 内的 `imagePath` 字段，其次按文件名 + 常见扩展名猜测。

---

## 4. copy_json_by_images.py

以图片文件名为基准，把同名的 JSON 标注文件从源目录批量拷贝到目标目录
（常用于“只保留有图片的那部分标注”）。

**用法**：

```bash
python copy_json_by_images.py \
    --img-dir /path/to/images \
    --json-src /path/to/json/source \
    --json-dst /path/to/json/dest
```

**参数**：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `--img-dir` | ✅ | 图片目录 |
| `--json-src` | ✅ | 源 JSON 目录 |
| `--json-dst` | ✅ | 目标 JSON 目录（自动创建） |
| `--recursive` | - | 递归遍历图片子目录 |
| `--ext` | - | 图片扩展名列表，默认 `.jpg .jpeg .png .bmp` |
| `--overwrite` | - | 覆盖目标已存在的 JSON（默认跳过） |
| `--dry-run` | - | 预览模式，不实际拷贝 |

**补充说明**：

- 按文件名（不含扩展名）匹配，目标目录不保留子目录结构。
- 使用 `shutil.copy2` 拷贝，保留文件元数据。

---

## 5. rename_xanylabeling_separate.py

批量重命名 X-AnyLabeling 数据集中的图片和 JSON（二者可位于不同目录），
重命名后**自动更新 JSON 内部的 `imagePath` 字段**。

支持三种互斥的重命名模式：

1. **通配符替换**：`--find "IMG_*" --replace "photo_\1"`
2. **前缀/后缀**：`--prefix new_` / `--suffix _v2`
3. **数字序列**：`--start-num 1`（可配合 `--prefix` / `--suffix`）

**用法示例**：

```bash
# 通配符替换：IMG_001.jpg -> photo_001.jpg
python rename_xanylabeling_separate.py \
    --img-dir /path/to/images \
    --json-dir /path/to/jsons \
    --find "IMG_*" \
    --replace "photo_\1"

# 统一加前缀
python rename_xanylabeling_separate.py \
    --img-dir /path/to/images --json-dir /path/to/jsons --prefix scene_

# 按数字序列从 1 开始重命名（4 位补零）
python rename_xanylabeling_separate.py \
    --img-dir /path/to/images --json-dir /path/to/jsons \
    --start-num 1 --seq-width 4 --prefix img_
```

**参数**：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `--img-dir` | ✅ | 图片目录 |
| `--json-dir` | ✅ | JSON 目录 |
| `--find` / `--replace` | 模式① | 通配符模式及替换串（`*` 匹配任意、`?` 匹配单字符，替换串可用 `\1` 引用） |
| `--prefix` | 模式②③ | 添加的前缀 |
| `--start-num` | 模式③ | 起始编号，启用数字序列模式 |
| `--suffix` | - | 添加的后缀（扩展名前），仅非通配符模式有效 |
| `--seq-width` | - | 数字序列位数，默认 4 |
| `--recursive` | - | 递归遍历图片子目录（JSON 仅在根目录查找） |
| `--force` | - | 覆盖已存在的目标文件（默认跳过） |
| `--dry-run` | - | 预览模式，不实际修改 |

**补充说明**：

- `--find` 与 `--replace` 必须成对出现；三者（通配符 / 前缀 / 数字序列）互斥。
- JSON 与图片必须**同名**（位于 `--json-dir` 根目录）才会被处理。
- 图片扩展名支持 `.jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp`。

---

## 6. fix_label.py

批量修正/替换 JSON 标注字段（大小写敏感），常见场景：

- 把 `imagePath` 里的 `.jpg` 后缀批量改成 `.jpeg`；
- 统一标签大小写（如 `Person` → `person`）；
- 递归替换任意自定义字段。

**用法示例**：

```bash
# 修正 imagePath 后缀 .jpg -> .jpeg（默认动作）
python fix_label.py /path/to/jsons

# 替换标签（递归，精确匹配，大小写敏感）
python fix_label.py /path/to/jsons --label-old Person --label-new person

# 替换自定义字段并禁用 imagePath 修正
python fix_label.py /path/to/jsons \
    --key group_id --old-value 3 --new-value 5 --no-image-fix

# 调试模式：只查找打印，不改动任何文件
python fix_label.py /path/to/jsons --label-old Person --debug
```

**参数**：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `directory` | - | 目标目录（默认当前目录，递归查找全部 JSON） |
| `--no-image-fix` | - | 禁用 imagePath 后缀修正（`.jpg` → `.jpeg`） |
| `--label-old` / `--label-new` | 成对 | 递归替换 label 字段的旧值/新值 |
| `--key` / `--old-value` / `--new-value` | 三个同用 | 递归替换自定义字段 |
| `--debug` | - | 调试模式：只查找并打印，不修改文件 |

**补充说明**：

- `directory` 会递归遍历其下所有 `.json` 文件。
- 字段替换为**精确匹配且大小写敏感**，需替换的分支才会被写入文件（有改动才写盘）。
- imagePath 修正仅作用于顶层字段且仅处理以 `.jpg` 结尾的情况。

---

## 7. sync_left_to_right.py

以**左侧文件夹为基准**，删除右侧文件夹中“多余”的文件（忽略扩展名比对），
常用于同步两份内容类似的数据集目录。

**用法**：

```bash
# 预览将要删除的文件（推荐先执行）
python sync_left_to_right.py <left_folder> <right_folder> --dry-run

# 实际执行（交互式确认）
python sync_left_to_right.py <left_folder> <right_folder>

# 跳过交互确认直接删除
python sync_left_to_right.py <left_folder> <right_folder> -y
```

**参数**：

| 参数 | 必选 | 说明 |
| --- | --- | --- |
| `left_folder` | ✅ | 左侧基准文件夹（不受影响） |
| `right_folder` | ✅ | 右侧待清理文件夹（多余文件会被删除） |
| `--dry-run` | - | 模拟运行，只打印不删除 |
| `--yes` / `-y` | - | 自动确认，跳过交互提示 |

**补充说明**：

- 比对规则：忽略扩展名的“相对路径”。例如左侧存在 `a/001.jpg`，
  则右侧的 `a/001.png` 视为匹配；右侧独有的文件将被删除。
- 递归处理所有子目录；删除前默认会交互确认。

---

## 常见工作流示例

**将 YOLO 数据集转成 X-AnyLabeling 并只导出指定类别**

```bash
# 1) YOLO -> X-AnyLabeling
python yolo_to_x_anylabeling.py \
    --img-dir images --txt-dir labels --json-out json_all \
    --classes-yaml data.yaml

# 2) 按标签导出子集
python export_by_labels.py \
    --src-json json_all --src-images images \
    --dst-json json_filtered --dst-images images_filtered \
    --labels person
```

**导入前统一命名规范**

```bash
# 预览重命名，确认后去掉 --dry-run
python rename_xanylabeling_separate.py \
    --img-dir raw_images --json-dir raw_jsons \
    --prefix defect_ --start-num 1 --seq-width 5 --dry-run
```
