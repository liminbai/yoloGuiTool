import argparse
import fiftyone as fo
# 禁用 FiftyOne / ETA 的严格包元数据检测（必须在加载 brain/zoo 模块之前设置）
fo.config.requirement_error_level = 2
import fiftyone.brain as fob
from fiftyone import ViewField as F


def ensure_similarity_index(dataset, brain_key="img_sim", model="mobilenet-v2-imagenet-torch"):
    """检查或重建图像相似度索引"""
    
    # 检查索引是否存在且有效
    if brain_key in dataset.list_brain_runs():
        sim_index = dataset.load_brain_results(brain_key)
        if sim_index is not None:
            print(f"⚡ 已加载有效相似度索引 [{brain_key}]")
            return sim_index
        else:
            print(f"⚠️ 检测到损坏的索引 [{brain_key}]，正在自动清理重构...")
            dataset.delete_brain_run(brain_key)

    print(f"🔍 正在使用 [{model}] 提取特征并建立索引...")
    
    # 加上 num_workers=0 和 batch_size=16 彻底防止共享内存爆满
    sim_index = fob.compute_similarity(
        dataset,
        model=model,
        brain_key=brain_key,
        batch_size=16,
        num_workers=0
    )
    print("✅ 相似度索引建立完成！")
    return sim_index


def remove_duplicates(dataset_name: str, threshold: float = 0.96, tag_only: bool = True):
    """
    自动标记或删除高度重复样本
    """
    dataset = fo.load_dataset(dataset_name)
    brain_key = "img_sim"
    
    # 获取索引对象
    sim_index = ensure_similarity_index(dataset, brain_key=brain_key)

    # 执行重复样本查找
    try:
        sim_index.find_duplicates(thresh=threshold)
    except TypeError:
        sim_index.find_duplicates(threshold)

    # 兼容获取重复样本的 ID 列表/集合
    duplicate_ids = set()
    
    # 方式 1: 优先获取 FiftyOne 新版/特定 index 的 duplicate_ids 属性
    if hasattr(sim_index, "duplicate_ids") and sim_index.duplicate_ids:
        duplicate_ids = set(sim_index.duplicate_ids)
    
    # 方式 2: 兼容组列表模式 (duplicates)
    elif hasattr(sim_index, "duplicates") and sim_index.duplicates:
        raw_dups = sim_index.duplicates
        if isinstance(raw_dups, dict):
            # 如果是 {keep_id: [dup_id1, dup_id2]} 字典结构
            for dup_list in raw_dups.values():
                duplicate_ids.update(dup_list)
        elif isinstance(raw_dups, list):
            # 如果是 [[keep_id, dup_id1], ...] 列表结构
            for group in raw_dups:
                duplicate_ids.update(group[1:])

    if not duplicate_ids:
        print(f"🎉 在相似度阈值 {threshold} 下未发现重复样本。")
        return

    print(f"⚠️ 经过筛选，共找出 {len(duplicate_ids)} 张高度重复的图片样本。")

    dup_view = dataset.select(list(duplicate_ids))
    
    if tag_only:
        dup_view.tag_samples("duplicate")
        print("✅ 已成功为重复样本添加 'duplicate' 标签！在 FiftyOne App 侧边栏可通过 TAGS 过滤查看。")
    else:
        dataset.delete_samples(dup_view)
        print("🗑️ 已成功从数据集完全删除重复样本！")


def find_similar_to_image(dataset_name: str, sample_id_or_path: str, k: int = 10):
    """
    功能 2: Python 侧的 Sort by similarity（查找与指定图片最相似的 K 张图片）
    """
    dataset = fo.load_dataset(dataset_name)
    brain_key = "img_sim"
    ensure_similarity_index(dataset, brain_key=brain_key)

    # 兼容传入 ID 或 路径
    if "/" in sample_id_or_path or "\\" in sample_id_or_path:
        sample = dataset.get_field("filepath")
        matches = dataset.match(F("filepath") == sample_id_or_path)
        if len(matches) == 0:
            print(f"❌ 未找到路径为 {sample_id_or_path} 的样本")
            return
        target_id = matches.first().id
    else:
        target_id = sample_id_or_path

    # 使用 SDK 进行相似度排序，返回按相似度从高到低排列的 View
    similar_view = dataset.sort_by_similarity(target_id, k=k, brain_key=brain_key)

    print(f"\n🎯 与目标样本 [{target_id}] 最相似的前 {k} 张图片如下：")
    for i, sample in enumerate(similar_view, 1):
        print(f"  {i}. ID: {sample.id} | 路径: {sample.filepath}")

    # 启动轻量 App 专门预览这 K 张相似图
    session = fo.launch_app(similar_view)
    session.wait()


def analyze_box_distribution(dataset_name: str, min_area_pct: float = 0.01, max_aspect_ratio: float = 5.0):
    """
    功能 3: 标注框分布异常排查（极小目标与极端宽高比目标）
    """
    dataset = fo.load_dataset(dataset_name)
    
    print("📊 正在分析标注框尺寸与宽高比分布...")
    small_boxes_count = 0
    extreme_aspect_count = 0

    for sample in dataset:
        if not sample.ground_truth:
            continue

        for det in sample.ground_truth.detections:
            w, h = det.bounding_box[2], det.bounding_box[3]
            area_pct = w * h * 100
            aspect_ratio = w / (h + 1e-6)

            if area_pct < min_area_pct:
                small_boxes_count += 1
            if aspect_ratio > max_aspect_ratio or aspect_ratio < (1.0 / max_aspect_ratio):
                extreme_aspect_count += 1

    print("\n--- 📈 数据集标注框分布诊断报告 ---")
    print(f"1. 极小目标框 (占全图面积 < {min_area_pct}%): {small_boxes_count} 个")
    print(f"2. 极端宽高比框 (宽高比 > {max_aspect_ratio} 或 < {1/max_aspect_ratio:.2f}): {extreme_aspect_count} 个")
    print("---------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FiftyOne 开源版数据集质量诊断与去重工具")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--action", type=str, choices=["dedup", "search", "analyze"], required=True, 
                        help="执行的操作: dedup(去重), search(相似度搜索), analyze(分布分析)")
    
    # 额外参数
    parser.add_argument("--threshold", type=float, default=0.96, help="去重相似度阈值 (默认 0.96)")
    parser.add_argument("--target", type=str, help="search 模式下的目标 sample_id 或 filepath")
    parser.add_argument("--k", type=int, default=10, help="search 模式下返回的最相似样本数")

    args = parser.parse_args()

    if args.action == "dedup":
        remove_duplicates(args.dataset, threshold=args.threshold, tag_only=True)
    elif args.action == "search":
        if not args.target:
            print("❌ 错误：使用 search 功能时必须指定 --target (sample_id 或 图片路径)")
        else:
            find_similar_to_image(args.dataset, args.target, k=args.k)
    elif args.action == "analyze":
        analyze_box_distribution(args.dataset)