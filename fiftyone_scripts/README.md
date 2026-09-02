# FiftyOne 数据集导入与增量同步说明

本目录用于管理 FiftyOne 数据集的创建、删除、导入和增量更新逻辑，专门支持 X-AnyLabeling 导出的 JSON 标注。

## 目录说明

- `anylabeling_import.py`  
  核心导入模块，负责：
  - 创建或加载数据集
  - 删除指定数据集
  - 解析 X-AnyLabeling JSON 标注
  - 将标注转换成 `fo.Detection`
  - 仅增量导入未存在的图片，避免重复导入

- `00_import_anylabeling.py`  
  初始导入入口。适合第一次把图片和对应 JSON 标注一起导入到数据集。

- `01_append_data.py`  
  增量追加入口。适合后续追加新图片和新标注，且不会重复导入已有数据。

- `dataLoad.py`  
  统一管理入口，用于：
  - 创建 dataset
  - 删除 dataset
  - 启动 FiftyOne 可视化界面
  - 直接指定图片目录和标注目录进行导入

---

## 目标功能

本套脚本实现以下能力：

1. 可手动创建和删除指定 dataset
2. 可导入图片和 X-AnyLabeling 标注
3. 每次执行时只新增未存在的数据，不重复导入
4. 标注字段写入 `ground_truth`
5. 支持与 FiftyOne 视图/YOLO 导出流程衔接

---

## 运行环境要求

当前项目中的 FiftyOne 实际安装在容器内的虚拟环境中，路径为：

```bash
/opt/.fiftyone-venv/bin/python
```

请务必使用这个 Python 解释器执行脚本，而不是宿主机的 `python3`。

---

## 常用命令

### 1. 创建数据集

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/dataLoad.py --create-dataset ppe_dataset
```

### 2. 删除数据集

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/dataLoad.py --delete-dataset ppe_dataset
```

### 3. 初次导入图片和 X-AnyLabeling 标注

`00_import_anylabeling.py` 支持命令行参数，常用参数如下：

| 参数 | 说明 |
| --- | --- |
| `--dataset-name` | 数据集名称（默认 `ppe_dataset`） |
| `--image-dir` | 源图片目录 |
| `--labels-dir` | X-AnyLabeling JSON 标注目录 |
| `-t/--tags` | 附加标签，可多个，如 `--tags raw_import incremental` |
| `--overwrite` | 数据集已存在时先删除再全量导入（清空脏数据） |

带参数直接运行：

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/00_import_anylabeling.py \
  --dataset-name ppe_dataset \
  --image-dir /media/images/ppe \
  --labels-dir /media/images/ppe_xany \
  --tags raw_import \
  --overwrite
```

不带参数运行会进入交互输入模式，逐个询问图片目录、标注目录与标签；也可通过 `-h` 查看全部帮助：

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/00_import_anylabeling.py
```

### 4. 增量追加新图片与新标注

`01_append_data.py` 同样支持命令行参数（与 `00` 入口一致），区别在于默认标签为 `raw_import incremental`，且完成提示为"增量导入完成"。

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/01_append_data.py \
  --dataset-name ppe_dataset \
  --image-dir /media/images/ppe \
  --labels-dir /media/images/ppe_xany \
  --tags raw_import incremental
```

不带参数运行会进入交互输入模式，逐个询问图片目录、标注目录与标签；也可通过 `-h` 查看全部帮助：

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/01_append_data.py
```

### 5. 启动 FiftyOne 可视化界面

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/dataLoad.py --dataset-name ppe_dataset --launch
```

### 6. 直接指定目录导入

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/dataLoad.py \
  --dataset-name ppe_dataset \
  --image-dir /media/images/ppe \
  --labels-dir /media/images/ppe_xany
```

### 7. 数据集质量诊断工具

`dataset_quality_tools.py` 位于当前目录，主要用于数据集清洗和质量检查，包含：

- `dedup`：根据图像相似度检测重复样本
- `search`：查找与指定样本最相似的图片
- `analyze`：分析极小目标与极端宽高比标注框分布

#### 7.1 去重

```bash
docker exec -it fo-dashboard \
  /opt/.fiftyone-venv/bin/python \
  /scripts/dataset_quality_tools.py \
  --dataset ppe_dataset \
  --action dedup \
  --threshold 0.96
```

说明：
- `--threshold` 可调节重复样本判定的相似度阈值。
- 默认行为是添加 `duplicate` 标签，便于在 FiftyOne App 中筛选查看。

#### 7.2 相似图搜索

```bash
docker exec -it fo-dashboard \
  /opt/.fiftyone-venv/bin/python \
  /scripts/dataset_quality_tools.py \
  --dataset ppe_dataset \
  --action search \
  --target /media/images/ppe/image_001.jpg \
  --k 10
```

也可传入 sample_id：

```bash
docker exec -it fo-dashboard \
  /opt/.fiftyone-venv/bin/python \
  /scripts/dataset_quality_tools.py \
  --dataset ppe_dataset \
  --action search \
  --target 1234567890abcdef \
  --k 10
```

#### 7.3 标注框分布分析

```bash
docker exec -it fo-dashboard \
  /opt/.fiftyone-venv/bin/python \
  /scripts/dataset_quality_tools.py \
  --dataset ppe_dataset \
  --action analyze
```

该功能会统计：
- 极小目标框（占全图面积很小）
- 极端长宽比框（如过宽或过高）

#### 7.4 使用建议

1. 先执行 `analyze`，快速发现明显的标注异常。
2. 再执行 `dedup`，过滤高度重复样本。
3. 对可疑样本执行 `search`，查看相似图片以确认是否存在误标或重复。

---

## 数据目录约定

本脚本默认按以下结构工作：

```text
/media/images/ppe          # 图片目录
/media/images/ppe_xany     # X-AnyLabeling JSON 标注目录
```

命名规则：

- 图片：`image1.jpeg`
- 标注：`image1.json`

如果图片和 JSON 文件名对应，脚本会自动匹配并导入。

---

## 增量导入机制说明

每次导入前，脚本都会检查当前数据集里已有的图片文件路径：

- 已存在则跳过
- 不存在则追加
- 已有 `ground_truth` 的样本也跳过重复写入

因此，重复执行脚本不会导致重复导入，符合“只增量添加”的要求。

---

## 标注转换说明

X-AnyLabeling 的 JSON 中，通常包含如下结构：

```json
{
  "imageWidth": 640,
  "imageHeight": 640,
  "shapes": [
    {
      "label": "helmet",
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    }
  ]
}
```

脚本会把每个矩形框转换成：

```python
fo.Detection(
    label="helmet",
    bounding_box=[x, y, w, h]
)
```

并存储到 `ground_truth` 字段中，便于后续在 FiftyOne 中查看和导出训练集。

---

## 常见注意事项

1. 必须使用容器内的虚拟环境 Python 执行脚本
2. 图片目录和标注目录必须存在
3. JSON 文件名必须和对应图片同名
4. 若数据集已存在，脚本会继续加载而不是重建
5. 若需要删除旧数据集，请显式执行删除命令

---

## 适合的使用流程

### 第一次使用

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/dataLoad.py --create-dataset ppe_dataset
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/00_import_anylabeling.py
```

### 后续追加新数据

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/01_append_data.py
```

### 查看数据集

```bash
docker exec -it fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/dataLoad.py --dataset-name ppe_dataset --launch
```

---

## 结论

这套脚本已经覆盖了你当前的核心需求：

- 手动管理数据集
- 导入 X-AnyLabeling 标注
- 自动绑定图片和标签
- 增量导入且不重复

可直接用于后续数据清洗、可视化查看和 YOLO 导出流程。
