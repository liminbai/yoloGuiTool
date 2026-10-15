# AI To Yolo GUI

  本项目通过本地客户端界面的方式将ultralytics工程主流的模型应用做了可视化应用配置，希望能给喜欢YOLO的同学提供多一种选择。该程序几项特点：

- 1）基于最新的PySide6图形界面进行了搭建；
- 2）主要支持（YOLOv8、YOLOv11、YOLOv26）这三个版本训练和推理；
- 3）支持对最新版本SAM3的推理；
- 4）集成了labelImg标注工具，升级到PySide6界面，并修复了本地化、删除图片跳转索引等问题。
- 5）操作区提供「📝 打开LabelImg」与「🏷 打开Anylabeling」按钮，可直接启动labelImg或通过 `xanylabeling` 命令启动X-AnyLabeling进行数据标注。
- 6）操作区提供「🔍 相似图像处理」按钮，内置仿 [AntiDupl](https://github.com/ermig1979/AntiDupl) 的相似/重复图片查找窗口，**仅保留 SSIM 比较**（参数对齐 AntiDuplX 的 `-cs` 比较尺寸与 `-ct` 比较阈值），用于清理数据集中重复或高度相似的图片。

![alt text](./docs/image.png)
![alt text](./docs/image-1.png)
![alt text](./docs/image-2.png)
![alt text](./docs/image-3.png)
![alt text](./docs/image-4.png)

- 主要文件
  - `yoloGui.py` — 兼容入口（保留 `python yoloGui.py` 启动方式）。
  - `gui/windows/trainer_main_window.py` — 主窗口与应用主流程（`YOLOTrainerGUI`、`main`）。
  - `gui/threads/yolo_training_thread.py` — 训练线程实现（`YOLOTrainingThread`）。
  - `gui/threads/yolo_inference_threads.py` — 推理线程实现（`YOLOInferenceThread`、`SAM3InferenceThread`）。
  - `gui/widgets/class_editor_dialog.py` — 类别编辑对话框（`ClassEditorDialog`）。
  - `gui/utils/image_similarity.py` — SSIM 相似度算法与相似图片对搜索（纯 numpy + Pillow 实现，无需 OpenCV）。
  - `gui/threads/similar_image_thread.py` — 相似图片搜索线程（`SimilarImageScanThread`）。
  - `gui/widgets/similar_image_window.py` — 相似图像处理窗口（仿 AntiDupl 主界面，仅 SSIM）。
  - `labelImg.py` — 原生图像标注界面。
  - `requirements-full.txt` — 完整依赖包列表（推荐安装）。

- 特性
  - 支持选择 YOLO 版本（`YOLOv8` / `YOLOv11` / `YOLOv26`）。
  - 支持任务类型：检测（detect）、分割（segment）、分类（classify）。
  - 根据版本与任务动态更新模型类型下拉列表（如 `yolov8n`、`yolov11m-seg` 等）。
  - 支持使用预训练权重或基于 YAML 配置生成新模型。
  - 包含训练线程（`YOLOTrainingThread`），会与 ultralytics YOLO API 对接执行训练并通过信号回传日志与进度。
  - 相似图像处理：按 AntiDupl 思路把图片缩放到统一比较尺寸（16/32/64/128）后计算 SSIM，列出相似度不低于阈值的图片对，支持双缩略图预览、删除（回收站/永久）与结果导出。

- 依赖
  - Python 3.8+
  - `PySide6` - GUI框架
  - `ultralytics>=8.2.0` - YOLO模型库（支持YOLOv8/v11/v26）
  - `torch>=2.1.0` & `torchvision>=0.16.0` - PyTorch深度学习框架
  - `opencv-python` - 图像处理
  - `pyyaml` - 配置文件处理
  - `pillow` - 图片解码与缩放（「🔍 相似图像处理」的 SSIM 比较依赖，算法的其余部分只用 numpy）
  - 可选依赖：
    - `segment-anything` - Meta原始SAM模型支持
    - `open_clip_torch` - CLIP文字提示支持
    - `transformers` - HuggingFace模型支持
    - `anylabeling` - X-AnyLabeling标注工具（安装后提供 `xanylabeling` 命令，「🏷 打开Anylabeling」按钮依赖该命令）

- 快速运行
  1. 激活或创建合适的 Python 环境，并安装依赖：

```bash
# 创建新的conda环境（推荐）
conda create -n yolo-gui python=3.10 -y
conda activate yolo-gui

# 安装核心依赖
pip install PySide6 ultralytics>=8.2.0 torch>=2.1.0 torchvision>=0.16.0 opencv-python pyyaml

# 安装可选依赖（用于SAM3文字提示功能）
pip install segment-anything open_clip_torch transformers
```

  或使用一键安装脚本：

```bash
# 克隆项目后运行
git clone <repository-url>
cd yoloGuiTool

# 自动安装所有依赖
pip install -r requirements-full.txt  # 推荐：一键安装所有依赖
```

1. 在项目目录下运行 GUI：

```bash
python yoloGui.py
```

## 注意

- 请确保 `ultralytics` 与 `PySide6` 可用于当前环境。
- `yoloGui.py` 中使用的模型名称与版本（例如 `yolov11`）应与 ultralytics/本地权重命名一致。
- SAM3功能需要PyTorch 2.1.0+和Ultralytics 8.2.0+支持。
- 文字提示功能需要安装可选依赖：`open_clip_torch` 或 `transformers`。

## 架构演进

- 若你准备拆分 `yoloGui.py`，可参考 `docs/yolo_gui_refactor_plan.md`。
