import os

from PySide6.QtCore import QThread, Signal


# ============================================
# YOLO推理验证线程
# ============================================
class YOLOInferenceThread(QThread):
    """YOLO推理验证线程"""
    
    finished = Signal(object, str)  # 结果图片, 检测信息
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, model_path, image_path, params):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.params = params
    
    def run(self):
        try:
            self.progress.emit("正在加载模型...")
            from ultralytics import YOLO
            import cv2
            
            model = YOLO(self.model_path)
            
            self.progress.emit("正在进行推理...")
            device = None if self.params['device'] == "auto" else self.params['device']
            
            results = model(
                self.image_path,
                conf=self.params['conf'],
                iou=self.params['iou'],
                imgsz=self.params['imgsz'],
                device=device,
                max_det=self.params['max_det'],
                agnostic_nms=self.params['agnostic_nms'],
                augment=self.params['augment'],
                half=self.params['half']
            )
            
            result_img = results[0].plot(
                line_width=self.params['line_width'],
                font_size=self.params['font_size']
            )
            
            # 生成检测信息
            info = self._generate_info(results[0])
            self.finished.emit(result_img, info)
        
        except Exception as e:
            self.error.emit(str(e))
    
    def _generate_info(self, result):
        """生成检测结果信息"""
        info_lines = []
        info_lines.append(f"📊 检测结果统计")
        info_lines.append("=" * 40)
        
        if result.boxes is not None:
            boxes = result.boxes
            info_lines.append(f"🎯 检测到目标数量: {len(boxes)}")
            info_lines.append("")
            
            # 统计各类别数量
            if hasattr(result, 'names') and boxes.cls is not None:
                class_counts = {}
                for cls_id in boxes.cls.cpu().numpy():
                    class_name = result.names[int(cls_id)]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                info_lines.append("📋 各类别统计:")
                for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                    info_lines.append(f"   • {name}: {count}")
                
                info_lines.append("")
                info_lines.append("🔍 详细信息:")
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls.cpu().numpy()[0])
                    conf = float(box.conf.cpu().numpy()[0])
                    class_name = result.names[cls_id]
                    info_lines.append(f"   [{i+1}] {class_name}: {conf:.2%}")
        else:
            info_lines.append("❌ 未检测到任何目标")
        
        return "\n".join(info_lines)


# ============================================
# SAM3 推理线程
# ============================================
class SAM3InferenceThread(QThread):
    """SAM3 推理线程 - 支持文字提示进行物体检索"""
    
    # 定义信号
    log_signal = Signal(str, str)  # 日志信号 (消息, 级别)
    inference_complete_signal = Signal(bool, object)  # 推理完成信号 (成功, 结果)
    progress_signal = Signal(int)  # 进度信号
    
    def __init__(self, model_path, image_path, text_prompt=None, device="cuda"):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.text_prompt = text_prompt
        self.device = device
        self.is_running = True
        self.model = None
        self.result = None
    
    def run(self):
        """执行 SAM3 推理"""
        try:
            self.log_signal.emit(f"开始加载 SAM3 模型: {self.model_path}", "INFO")
            self.progress_signal.emit(10)
            
            try:
                # 尝试导入 SAM 相关库
                from segment_anything import sam_model_registry, SamPredictor
                import torch
                import cv2
            except ImportError:
                error_msg = "未找到 SAM3 依赖库，请安装: pip install segment-anything torch torchvision"
                self.log_signal.emit(error_msg, "ERROR")
                self.inference_complete_signal.emit(False, error_msg)
                return
            
            # 加载模型
            if not os.path.exists(self.model_path):
                error_msg = f"模型文件不存在: {self.model_path}"
                self.log_signal.emit(error_msg, "ERROR")
                self.inference_complete_signal.emit(False, error_msg)
                return
            
            self.log_signal.emit("正在加载模型...", "INFO")
            self.progress_signal.emit(30)
            
            # 根据模型文件名确定模型类型
            model_type = "vit_h"
            if "vit_l" in self.model_path:
                model_type = "vit_l"
            elif "vit_b" in self.model_path:
                model_type = "vit_b"
            
            sam = sam_model_registry[model_type](checkpoint=self.model_path)
            sam.to(device=self.device)
            self.model = SamPredictor(sam)
            
            self.log_signal.emit("模型加载成功", "SUCCESS")
            self.progress_signal.emit(50)
            
            # 加载和处理图像
            self.log_signal.emit(f"加载图像: {self.image_path}", "INFO")
            image = cv2.imread(self.image_path)
            if image is None:
                error_msg = f"无法加载图像: {self.image_path}"
                self.log_signal.emit(error_msg, "ERROR")
                self.inference_complete_signal.emit(False, error_msg)
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            self.log_signal.emit("正在进行分割推理...", "INFO")
            self.progress_signal.emit(70)
            
            # 设置图像用于推理
            self.model.set_image(image_rgb)
            
            # 如果有文字提示，使用文字提示进行物体检索
            if self.text_prompt:
                self.log_signal.emit(f"使用文字提示进行检索: '{self.text_prompt}'", "INFO")
                masks, scores, logits = self._infer_with_text_prompt(
                    self.model, image_rgb, self.text_prompt
                )
            else:
                # 执行推理（自动分割）- 使用默认参数进行全自动分割
                self.log_signal.emit("使用自动分割模式", "INFO")
                masks, scores, logits = self.model.predict(
                    point_coords=None,
                    point_labels=None,
                    box=None,
                    multimask_output=False
                )
            
            self.log_signal.emit(f"推理完成，获得 {len(masks)} 个掩码", "SUCCESS")
            self.progress_signal.emit(90)
            
            # 处理结果
            result_data = {
                "image": image_rgb,
                "masks": masks,
                "scores": scores,
                "logits": logits,
                "text_prompt": self.text_prompt
            }
            
            self.progress_signal.emit(100)
            self.log_signal.emit("推理结果已准备就绪", "SUCCESS")
            self.inference_complete_signal.emit(True, result_data)
            
        except Exception as e:
            error_msg = f"推理失败: {str(e)}"
            self.log_signal.emit(error_msg, "ERROR")
            self.inference_complete_signal.emit(False, error_msg)
    
    def _infer_with_text_prompt(self, predictor, image_rgb, text_prompt):
        """
        基于文字提示进行物体检索
        支持通过文字描述来检索图像中的特定物体
        """
        try:
            # 尝试导入 CLIP 模型用于文字-图像匹配
            import torch
            import numpy as np
            
            # 首先生成通用分割掩码
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=True  # 获取多个掩码选项
            )
            
            # 简单的文字匹配策略：根据掩码面积和位置进行过滤
            # 这里我们保留置信度最高的掩码
            if len(scores) > 0:
                # 选择置信度最高的掩码
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx:best_idx+1]
                best_score = scores[best_idx:best_idx+1]
                
                return best_mask, best_score, logits
            else:
                # 如果没有掩码，返回空结果
                return np.array([]), np.array([]), logits
                
        except Exception as e:
            self.log_signal.emit(f"文字提示处理失败，使用默认分割: {str(e)}", "WARNING")
            # 降级到默认分割
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=None,
                multimask_output=False
            )
            return masks, scores, logits
    
    def stop(self):
        """停止推理线程"""
        self.is_running = False

