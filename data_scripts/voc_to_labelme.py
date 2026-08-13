import os
import json
import xml.etree.ElementTree as ET
import argparse  # 新增命令行解析模块

def voc_to_labelme(xml_dir, json_output_dir):
    """
    将 Pascal VOC XML 格式批量转换为 LabelMe / X-AnyLabeling JSON 格式
    """
    if not os.path.exists(xml_dir):
        print(f"错误：输入目录 '{xml_dir}' 不存在")
        return

    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)

    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    if not xml_files:
        print(f"警告：在 '{xml_dir}' 中未找到任何 .xml 文件")
        return

    for xml_file in xml_files:
        xml_path = os.path.join(xml_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            print(f"跳过 '{xml_file}'：XML 解析失败")
            continue

        # 读取图像名称与尺寸
        filename = root.find('filename')
        if filename is None or filename.text is None:
            print(f"跳过 '{xml_file}'：缺少 <filename> 字段")
            continue
        filename = filename.text

        size = root.find('size')
        if size is None:
            print(f"跳过 '{xml_file}'：缺少 <size> 字段")
            continue
        width_elem = size.find('width')
        height_elem = size.find('height')
        if width_elem is None or height_elem is None:
            print(f"跳过 '{xml_file}'：缺少 <width> 或 <height> 字段")
            continue
        width = int(width_elem.text)
        height = int(height_elem.text)

        # 构建 LabelMe JSON 结构
        labelme_data = {
            "version": "0.4.0",
            "flags": {},
            "shapes": [],
            "imagePath": filename,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width
        }

        # 解析标注对象
        for obj in root.findall('object'):
            label_name = obj.find('name')
            if label_name is None or label_name.text is None:
                continue
            label_name = label_name.text

            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            # 获取坐标并转换为 float
            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
            except (ValueError, AttributeError):
                print(f"警告：在 '{xml_file}' 中跳过无效的边界框")
                continue

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

        # 保存为对应名称的 JSON 文件
        json_filename = os.path.splitext(xml_file)[0] + '.json'
        json_path = os.path.join(json_output_dir, json_filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

    print(f"转换完成，已处理 {len(xml_files)} 个 VOC XML 标注文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 Pascal VOC XML 标注批量转换为 LabelMe / X-AnyLabeling JSON 格式"
    )
    parser.add_argument(
        "xml_dir",
        help="包含 VOC XML 文件的输入目录"
    )
    parser.add_argument(
        "json_output_dir",
        help="保存生成的 JSON 文件的输出目录（如果不存在会自动创建）"
    )
    # 可选参数：是否覆盖已有文件（默认跳过），为了简单暂不添加
    args = parser.parse_args()

    voc_to_labelme(args.xml_dir, args.json_output_dir)