#!/usr/bin/env python3
"""
YOLO转换功能测试脚本
测试ONNX和TensorRT转换功能
"""

import os
import sys
import tempfile
from pathlib import Path

def test_onnx_conversion():
    """测试ONNX转换功能"""
    try:
        from ultralytics import YOLO
        print("测试ONNX转换...")

        # 创建临时目录用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            # 下载一个小的YOLOv8模型进行测试
            model = YOLO('yolov8n.pt')  # 这会自动下载模型
            print(f"模型加载成功: {model}")

            # 设置输出路径
            output_path = os.path.join(temp_dir, 'test_model.onnx')

            # 执行ONNX导出
            print("开始ONNX导出...")
            success = model.export(
                format="onnx",
                imgsz=640,
                opset=17,
                save_dir=temp_dir,
                name='test_model'
            )

            if success:
                print("✅ ONNX转换成功!")
                # 检查文件是否存在
                if os.path.exists(output_path):
                    print(f"输出文件: {output_path}")
                    print(f"文件大小: {os.path.getsize(output_path)} bytes")
                else:
                    print("❌ 输出文件不存在")
                    return False
            else:
                print("❌ ONNX转换失败")
                return False

        return True

    except Exception as e:
        print(f"❌ ONNX转换测试失败: {e}")
        return False

def test_tensorrt_conversion():
    """测试TensorRT转换功能"""
    try:
        from ultralytics import YOLO
        print("\n测试TensorRT转换...")

        # 检查是否有CUDA
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA不可用，跳过TensorRT测试")
            return True

        # 创建临时目录用于测试
        with tempfile.TemporaryDirectory() as temp_dir:
            # 加载模型
            model = YOLO('yolov8n.pt')
            print(f"模型加载成功: {model}")

            # 设置输出路径
            output_path = os.path.join(temp_dir, 'test_model.engine')

            # 执行TensorRT导出
            print("开始TensorRT导出...")
            success = model.export(
                format="engine",
                imgsz=640,
                half=True,  # FP16
                save_dir=temp_dir,
                name='test_model'
            )

            if success:
                print("✅ TensorRT转换成功!")
                # 检查文件是否存在
                if os.path.exists(output_path):
                    print(f"输出文件: {output_path}")
                    print(f"文件大小: {os.path.getsize(output_path)} bytes")
                else:
                    print("❌ 输出文件不存在")
                    return False
            else:
                print("❌ TensorRT转换失败")
                return False

        return True

    except Exception as e:
        print(f"❌ TensorRT转换测试失败: {e}")
        return False

if __name__ == "__main__":
    print("YOLO转换功能测试")
    print("=" * 50)

    # 测试ONNX转换
    onnx_success = test_onnx_conversion()

    # 测试TensorRT转换
    tensorrt_success = test_tensorrt_conversion()

    print("\n" + "=" * 50)
    print("测试结果:")
    print(f"ONNX转换: {'✅ 通过' if onnx_success else '❌ 失败'}")
    print(f"TensorRT转换: {'✅ 通过' if tensorrt_success else '❌ 失败'}")

    if onnx_success and tensorrt_success:
        print("\n🎉 所有测试通过!")
    else:
        print("\n⚠️  部分测试失败，请检查依赖和环境配置")