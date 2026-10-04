# FiftyOne 数据集导入与增量同步说明

本目录用于管理 FiftyOne 数据集的创建、删除、导入和增量更新逻辑，专门支持 X-AnyLabeling 导出的 JSON 标注。

## 目录说明

- `anylabeling_import.py`  
  核心导入模块，负责：
  - 创建或加载数据集
  - 删除指定数据集
  - 解析 X-AnyLabeling JSON 标注
  - `rectangle` 转换成 `fo.Detection`（写入 `ground_truth`），
    `polygon` / `rotation` 额外转换成 `fo.Polyline`（写入 `ground_truth_polygons`）
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

- `02_export_yolo.py`  
  导出入口，用于：
  - 按 train/val 随机拆分导出 `yolo`（检测，ultralytics 可直接 `yolo detect train`）
  - 导出 `yolo-seg`（多边形，ultralytics 可直接 `yolo segment train`）
  - 自动生成 `dataset.yaml` / `data.yaml` 训练配置

---

## 目标功能

本套脚本实现以下能力：

1. 可手动创建和删除指定 dataset
2. 可导入图片和 X-AnyLabeling 标注
3. 每次执行时只新增未存在的数据，不重复导入
4. 标注字段写入 `ground_truth`（矩形框，`Detections`）；
   多边形/旋转框轮廓写入 `ground_truth_polygons`（`Polylines`，归一化坐标、闭合填充）
5. 支持与 FiftyOne 视图/YOLO 导出流程衔接
6. 可用 `--refresh-labels` 重新解析全部 JSON 并覆盖标注（补齐/修正标注）

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
| `--refresh-labels` | 忽略已有标注，重新解析所有样本 JSON 并覆盖写入（用于补齐 `ground_truth_polygons` 或同步标注修改） |

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

### 4.1 补齐 / 刷新多边形标注

若数据集是早期版本（只有 `ground_truth` 矩形框）导入的，加 `--refresh-labels` 重新解析全部 JSON，
即可回填 `ground_truth_polygons` 多边形字段（已存在的直角框会被重写为最新 JSON 内容）：

```bash
docker exec fo-dashboard /opt/.fiftyone-venv/bin/python /scripts/00_import_anylabeling.py \
  --dataset-name Puddle \
  --image-dir /media/images/Puddle \
  --labels-dir /media/images/Puddle_xany \
  --refresh-labels
```

导入纯多边形数据集（如 `Puddle`，`shape_type` 全为 `polygon`）后，可在 App 中用
`ground_truth` 筛选矩形框视图、用 `ground_truth_polygons` 查看真实轮廓（`closed=True`、`filled=True`）。

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

## 8. 导出可直接训练的 YOLO 数据集（检测 / 实例分割）

`02_export_yolo.py` 负责把 FiftyOne 数据集按 train/val 拆分导出，并生成配置文件。

### 8.1 格式对照

| `--format` | 标注格式 | 用途 | 配置文件 | 默认标注字段 |
| --- | --- | --- | --- | --- |
| `yolo`（默认） | YOLO `.txt`（框 `<class> <xc> <yc> <w> <h>`） | ✅ `yolo detect train` | `dataset.yaml` | `ground_truth` |
| `yolo-seg` | YOLO-seg `.txt`（多边形 `<class> <x1> <y1> ... <xn> <yn>`） | ✅ `yolo segment train` | `data.yaml` | `ground_truth_polygons` |

> ultralytics 只读取与图片同级的 `labels/*.txt`；它通过把图片路径里的 `images`
> 替换成 `labels` 来定位标注，所以图片必须在 `images/<split>`、标注必须在
> `labels/<split>`。

### 8.2 导出

```bash
docker exec fo-dashboard \
  /opt/.fiftyone-venv/bin/python \
  /scripts/02_export_yolo.py \
  --dataset-name fire_dataset \
  --format yolo \
  --overwrite
```

导出结构：

```text
/exports/yolo/fire_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

### 8.3 开始训练

在宿主机上直接执行：

```bash
yolo detect train \
  data=./exports/yolo/fire_dataset/dataset.yaml \
  model=yolov8m.pt \
  epochs=100 imgsz=640 batch=16 device=0
```

`dataset.yaml` 中**不写** `path` 键，ultralytics 会以该 YAML 所在目录为基准解析
`train`/`val`，因此整份导出目录可以直接拷贝或移动到其它机器上使用。

### 8.4 实例分割导出（yolo-seg）

`--format yolo-seg` 从多边形字段 `ground_truth_polygons`（`Polylines`）导出 YOLO-seg 标签，
每行是 `<class> <x1> <y1> <x2> <y2> ... <xn> <yn>`（归一化多边形顶点）：

```bash
docker exec fo-dashboard \
  /opt/.fiftyone-venv/bin/python \
  /scripts/02_export_yolo.py \
  --dataset-name puddle_dataset \
  --format yolo-seg \
  --overwrite
```

默认导出目录是 `/exports/yolo_seg/<dataset_name>`，
结构为 `images/{train,val}` + `labels/{train,val}` + `data.yaml`。

```bash
yolo segment train \
  data=./exports/yolo_seg/puddle_dataset/data.yaml \
  model=yolov8m-seg.pt \
  epochs=100 imgsz=640 batch=16 device=0
```

注意事项（脚本已做前置检查，这里说明原理）：

- 数据集中必须有多边形标注。若只有 `ground_truth`（矩形框），脚本会直接报错并提示：
  用 `00_import_anylabeling.py --refresh-labels` 生成 `ground_truth_polygons` 后再导出。
- ultralytics 把「一行 tokens 数 > 6」的行判定为分割行，因此顶点数 < 3 的多边形会被剔除，
  没有任何可用多边形的样本不会进入导出（脚本会打印剔除数量）。
- 导出后脚本会自检每个 split 的图片/标签是否一一对应、每行是否合法，并把越界坐标
  裁剪到 `[0, 1]`（ultralytics 校验时坐标超出 `[-0.01, 1.01]` 会中止整个数据集）。

### 8.5 常用参数

| 参数 | 说明 |
| --- | --- |
| `--label-field` | 标注字段（`yolo` 默认 `ground_truth`，`yolo-seg` 默认 `ground_truth_polygons`） |
| `--classes` | 类别列表；不传则自动从数据集推断 |
| `--train-ratio` / `--val-ratio` | 拆分比例（默认 0.8 / 0.2） |
| `--seed` | 随机划分种子（默认 42） |
| `--max-samples` | 只导出随机 N 个样本，用于快速冒烟测试 |
| `--overwrite` | 导出目录已存在时先清空，避免残留上一版图片/标注 |
| `--skip-missing-media` | 跳过源图片缺失的样本，而不是中止导出 |

> 重新导出时务必带 `--overwrite`，否则旧的图片和标注会残留在目录里。
> `--splits` 必须同时包含 `train` 和 `val`，脚本会提前校验。

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
