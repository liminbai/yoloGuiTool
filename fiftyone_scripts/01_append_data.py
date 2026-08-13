import fiftyone as fo

dataset_name = "ppe_dataset"

# 1. 加载或新建数据集
if dataset_name in fo.list_datasets():
    dataset = fo.load_dataset(dataset_name)
    print(f"📂 已加载已有数据集 [{dataset_name}]，当前基线样本数: {len(dataset)}")
else:
    dataset = fo.Dataset(dataset_name)
    print(f"✨ 新建数据集 [{dataset_name}]")

# 2. 增量扫描并导入新增数据及 X-AnyLabeling (Labelme 格式兼容) 标注
# FiftyOne 会根据图片路径自动去重，只吸收新新增的图片与 JSON
dataset.add_dir(
    dataset_dir="/media/images/ppe",       # 映射的容器内路径
    dataset_type=fo.types.LabelmeDataset,  # AnyLabeling 输出格式兼容 Labelme
    labels_path="/media/images/ppe",       # JSON 文件所在路径
    label_field="ground_truth"              # 标注字段名称
)

# 3. 持久化更新
dataset.save()
print(f"✅ 增量导入完成！更新后数据集总样本数: {len(dataset)}")