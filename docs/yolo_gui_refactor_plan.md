# yoloGui.py 拆分建议（渐进式）

当前 `yoloGui.py` 已经包含：
- 多个业务线程（训练、推理、分割推理）
- 多个 QWidget/QDialog 视图
- 主窗口装配与菜单逻辑
- 配置装载与持久化逻辑

这类“单文件承载所有职责”的写法在早期迭代很快，但后续会明显增加维护成本（定位问题、合并冲突、复用困难、测试困难）。

## 推荐目录结构

```text
project_root/
  yoloGui.py                      # 仅作为兼容入口，尽量薄
  yolo_training_thread.py         # 训练线程（本次已抽离）
  gui/
    windows/
      trainer_main_window.py      # YOLOTrainerGUI
    widgets/
      yolo_config_widget.py       # YOLOConfigWidget
      training_monitor_widget.py  # TrainingMonitorWidget
      inference_widget.py         # YOLOInferenceWidget
      sam3_inference_widget.py    # SAM3InferenceWidget
      class_editor_dialog.py      # ClassEditorDialog
    styles/
      common.qss
  services/
    training_service.py
    inference_service.py
  models/
    app_config.py                 # dataclass/pydantic 统一配置模型
```

## 拆分原则

1. **先抽线程，再抽视图**：线程类相对独立，最容易先拆。
2. **主窗口只做组装**：`YOLOTrainerGUI` 只负责创建/连接子组件，不直接承载业务细节。
3. **配置统一模型化**：把 `dict` 迁移为结构化配置对象，减少键名拼写类错误。
4. **信号边界清晰**：跨模块通信通过 Qt Signal，不做“跨层直接访问子控件”。

## 建议的三阶段改造

### Phase 1（低风险）
- 抽离线程类：`YOLOTrainingThread`、`YOLOInferenceThread`、`SAM3InferenceThread`
- `yoloGui.py` 中仅保留 import 与连接逻辑

### Phase 2（中风险）
- 抽离 `ClassEditorDialog`、`YOLOConfigWidget`、`TrainingMonitorWidget`
- 将样式字符串搬到 `gui/styles/common.qss`

### Phase 3（中高风险）
- 增加 `services/` 承载与 ultralytics 的调用
- 增加 `models/app_config.py` 做配置校验和默认值管理
- 为关键逻辑添加最小单元测试

## 本次已完成

- 已将 `YOLOTrainingThread` 从 `yoloGui.py` 抽离到 `yolo_training_thread.py`。
- 已将 `YOLOInferenceThread` 与 `SAM3InferenceThread` 抽离到 `yolo_inference_threads.py`。
- 已将 `ClassEditorDialog` 抽离到 `class_editor_dialog.py`。
