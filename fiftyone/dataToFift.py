import fiftyone as fo

# 1. 加载或创建指定数据集
dataset_name = "ppe_dataset"
if dataset_name in fo.list_datasets():
    dataset = fo.load_dataset(dataset_name)
else:
    dataset = fo.Dataset(dataset_name)

# 2. 扫描共享挂载目录（自动对比并只加入新增图像）
# 注意：容器内路径必须一致使用 /media/images/...
dataset.add_images_dir(
    images_dir="/media/images/ppe",
    tags=["raw_import", "unlabeled"]  # 给导入的样本赋予初始标签
)

# 3. 持久化数据更改
dataset.save()
print(f"✅ 数据集 [{dataset_name}] 增量同步完成，当前总样本数: {len(dataset)}")