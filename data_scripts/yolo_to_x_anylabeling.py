import os
import json
import cv2

# 1. 映射类别 ID 到具体的类名（请根据你训练/标注时的 classes.txt 替换）
CLASSES = [
    "reflective_vest_half",
    "helmet",
    "person"
    # ... 在这里按顺序补充你的类别名称
]

def yolo_to_x_anylabeling(img_dir, txt_dir, json_out_dir, classes_list):
    """
    将 YOLO txt 标注转换为 X-AnyLabeling 兼容的 JSON 格式
    """
    if not os.path.exists(json_out_dir):
        os.makedirs(json_out_dir)

    # 常见图片格式后缀
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG')
    img_files = [f for f in os.listdir(img_dir) if f.endswith(valid_exts)]

    success_count = 0

    for img_name in img_files:
        base_name = os.path.splitext(img_name)[0]
        txt_name = f"{base_name}.txt"
        
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(txt_dir, txt_name)

        # 获取图片的真实宽高
        img = cv2.imread(img_path)
        if img is None:
            print(f"[警告] 无法读取图片: {img_path}，已跳过")
            continue
        
        img_height, img_width = img.shape[:2]

        # 初始化 X-AnyLabeling 标准结构
        labelme_data = {
            "version": "0.4.0",
            "flags": {},
            "shapes": [],
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width
        }

        # 如果对应的 txt 文件存在，读取标注信息
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # 映射类别名称
                label_name = classes_list[class_id] if class_id < len(classes_list) else str(class_id)

                # 归一化坐标还原为像素点坐标
                abs_w = w * img_width
                abs_h = h * img_height
                abs_cx = x_center * img_width
                abs_cy = y_center * img_height

                xmin = abs_cx - (abs_w / 2.0)
                ymin = abs_cy - (abs_h / 2.0)
                xmax = abs_cx + (abs_w / 2.0)
                ymax = abs_cy + (abs_h / 2.0)

                # 构建 X-AnyLabeling 绑定的 shape 对象
                shape = {
                    "label": label_name,
                    "score": None,
                    "points": [
                        [xmin, ymin],
                        [xmax, ymax]
                    ],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                    "direction": 0.0
                }
                labelme_data["shapes"].append(shape)

        # 写入生成 JSON 文件
        json_path = os.path.join(json_out_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        success_count += 1

    print(f"处理完成！成功生成 {success_count} 个 X-AnyLabeling JSON 标注文件。")

# ================= 脚本配置与运行 =================
if __name__ == "__main__":
    IMAGE_DIR = "./dataset/images"      # 图片存放路径
    TXT_DIR = "./dataset/labels"        # YOLO txt 存放路径
    JSON_OUT_DIR = "./dataset/jsons"    # 输出 JSON 的目标路径

    yolo_to_x_anylabeling(IMAGE_DIR, TXT_DIR, JSON_OUT_DIR, CLASSES)