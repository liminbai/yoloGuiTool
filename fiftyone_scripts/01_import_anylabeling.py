import fiftyone as fo

# 1. 创建或加载数据集
dataset_name = "ppe_dataset"
dataset = fo.Dataset(dataset_name) if dataset_name not in fo.list_datasets() else fo.load_dataset(dataset_name)

# 2. 导入图像及其关联的 X-AnyLabeling (JSON) 标注
# FiftyOne 会自动寻找与图片同名的 .json 标注文件
dataset.add_dir(
    dataset_dir="/media/images/ppe",
    dataset_type=fo.types.LabelmeDataset,  # AnyLabeling 输出格式兼容 Labelme
    labels_path="/media/images/ppe",       # JSON 文件所在的同一目录
    label_field="ground_truth"
)

dataset.save()
print(f"✅ 标注导入完成，当前数据集样本数: {len(dataset)}")