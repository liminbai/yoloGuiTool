# 配置说明

以下说明基于 `yoloGui.py` 中读取与传递到训练线程的关键配置项。GUI 会组合这些配置并最终传给 `YOLOTrainingThread`，然后传递给 `ultralytics.YOLO.train()`。

示例（JSON）:

```json
{
  "model": {
    "family": "yolov11",
    "type": "yolov11m",
    "pretrained": true
  },
  "training": {
    "epochs": 100
    // 可选：其他将会被传递给 ultralytics 的训练参数，如 batch、lr、imgsz 等（视具体实现）
  },
  "data": {
    "train": "/path/to/train",
    "val": "/path/to/val",
    "test": "/path/to/test"
  }
}
```

关键项说明
- `model.family`：字符串，表示模型系列，取值为 `yolov8`、`yolov11`、`yolov26`。
- `model.type`：字符串，表示模型变体，例如 `yolov8n`、`yolov11m-seg`、`yolov26x-cls` 等。GUI 的 `模型类型` 下拉框会提供有效选项。
- `model.pretrained`：布尔，是否使用预训练权重文件（`.pt`）。
- `training.epochs`：训练轮数，`YOLOTrainingThread` 中通过 `self.total_epochs` 读取。
- `data`：数据集路径（训练/验证/测试），GUI 提供路径选择控件。

备注
- `yoloGui.py` 在加载模型失败时，会尝试使用默认轻量模型（例如 `yolov8n.pt` / `yolov11n.pt` / `yolov26n.pt`）作为后备。
- GUI 会根据 `model.type` 的后缀自动推断任务类型（包含 `-seg` 或 `-cls`），以便在界面或训练参数中使用。

扩展
- 如果你打算从配置文件启动程序，可以把上面的 JSON/YAML 保存为 `config.json`/`config.yaml`，并在启动脚本中加载然后传入 GUI 或训练线程。