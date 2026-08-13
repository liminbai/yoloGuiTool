import fiftyone as fo
import fiftyone.utils.random as four

# 1. 加载包含标注的数据集
dataset = fo.load_dataset("ppe_dataset")
valid_view = dataset.exists("ground_truth")

# 2. 按 8:2 随机划分训练集与验证集
four.random_split(valid_view, {"train": 0.8, "val": 0.2})

# 3. 导出标准 YOLO 格式
export_dir = "/exports/yolo/ppe_dataset"
valid_view.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth",
    split=["train", "val"],
    classes=["helmet", "vest", "person"]  # 填入你的具体类别
)

print(f"🚀 YOLO 训练集已成功导出至: {export_dir}")