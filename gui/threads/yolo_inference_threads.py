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
    """SAM3 推理线程 - 仅支持Ultralytics SAM3模型，支持文字提示进行物体检索"""
    
    # 定义信号
    log_signal = Signal(str, str)  # 日志信号 (消息, 级别)
    inference_complete_signal = Signal(bool, object)  # 推理完成信号 (成功, 结果)
    progress_signal = Signal(int)  # 进度信号
    
    def __init__(self, model_path, image_path, text_prompt=None, device="cuda", clip_threshold=0.2, top_n=3):
        super().__init__()
        self.model_path = model_path
        self.image_path = image_path
        self.text_prompt = text_prompt
        self.device = device
        self.clip_threshold = clip_threshold
        self.top_n = top_n
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
                import torch
                import cv2
            except ImportError:
                error_msg = "未找到 SAM3 依赖库，请安装: pip install torch>=2.1.0 torchvision ultralytics>=8.2.0"
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
            model_filename = os.path.basename(self.model_path).lower()
            if "sm3" in model_filename or "sam3" in model_filename:
                # 检查版本兼容性
                try:
                    import torch
                    import ultralytics
                    torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
                    ultralytics_version = tuple(map(int, ultralytics.__version__.split('.')[:2]))

                    min_torch_version = (2, 1)
                    min_ultralytics_version = (8, 2)

                    version_ok = (torch_version >= min_torch_version and
                                ultralytics_version >= min_ultralytics_version)

                    if version_ok:
                        self.log_signal.emit(f"环境版本检查通过: PyTorch {torch.__version__}, Ultralytics {ultralytics.__version__}", "INFO")
                        # 使用Ultralytics SAM3/SM3模型
                        try:
                            from ultralytics import SAM
                            self.log_signal.emit(f"检测到SM3/SAM3模型 ({model_filename})，使用Ultralytics SAM加载方式", "INFO")
                            self.model = SAM(self.model_path)
                            self.model.to(device=self.device)
                            self.is_ultralytics_model = True
                            self.log_signal.emit("Ultralytics SAM3模型加载成功", "SUCCESS")
                        except Exception as e:
                            error_msg = f"Ultralytics SAM3加载失败: {str(e)}"
                            self.log_signal.emit(error_msg, "ERROR")
                            self.log_signal.emit("请检查:", "INFO")
                            self.log_signal.emit("1. 模型文件是否为Ultralytics格式的SAM3模型", "INFO")
                            self.log_signal.emit("2. PyTorch版本是否>=2.1.0", "INFO")
                            self.log_signal.emit("3. Ultralytics版本是否>=8.2.0", "INFO")
                            self.inference_complete_signal.emit(False, error_msg)
                            return
                    else:
                        error_msg = f"环境版本不足: PyTorch {torch.__version__} (需要>=2.1.0), Ultralytics {ultralytics.__version__} (需要>=8.2.0)"
                        self.log_signal.emit(error_msg, "ERROR")
                        self.log_signal.emit("请升级依赖版本后再使用SAM3功能", "INFO")
                        self.inference_complete_signal.emit(False, error_msg)
                        return

                except ImportError as e:
                    error_msg = f"缺少必要的依赖库: {str(e)}"
                    self.log_signal.emit(error_msg, "ERROR")
                    self.log_signal.emit("请安装: pip install torch>=2.1.0 ultralytics>=8.2.0", "INFO")
                    self.inference_complete_signal.emit(False, error_msg)
                    return
            else:
                # 不支持传统SAM模型
                error_msg = f"不支持的模型类型: {model_filename}"
                self.log_signal.emit(error_msg, "ERROR")
                self.log_signal.emit("SAM3推理仅支持Ultralytics格式的SAM3模型", "INFO")
                self.log_signal.emit("请使用sam3.pt或sm3.pt格式的模型文件", "INFO")
                self.inference_complete_signal.emit(False, error_msg)
                return
            
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
            
            if self.is_ultralytics_model:
                # 使用Ultralytics SAM3模型进行推理
                self.log_signal.emit("使用Ultralytics SAM3进行推理", "INFO")

                if self.text_prompt:
                    self.log_signal.emit(f"使用文字提示进行检索: '{self.text_prompt}'", "INFO")

                masks, scores, logits, boxes = self._infer_ultralytics_sam3(image_rgb, self.text_prompt)
            else:
                # 不应该到达这里，因为我们只支持Ultralytics SAM3
                error_msg = "不支持的模型类型"
                self.log_signal.emit(error_msg, "ERROR")
                self.inference_complete_signal.emit(False, error_msg)
                return

            self.log_signal.emit(f"推理完成，获得 {len(masks)} 个掩码", "SUCCESS")
            self.progress_signal.emit(90)
            
            # 处理结果
            result_data = {
                "image": image_rgb,
                "masks": masks,
                "scores": scores,
                "boxes": boxes,
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
    
    def _infer_with_text_prompt(self, predictor, image_rgb, text_prompt, ultralytics=False):
        """
        基于文字提示进行物体检索
        仅支持Ultralytics SAM3模型
        """
        try:
            import numpy as np
            import os

            if ultralytics:
                # Ultralytics SAM3使用predict()进行推理（不支持直接文本参数）
                # 文本匹配通过CLIP在得到所有候选后进行
                try:
                    # 调用predict()获得所有分割候选
                    if hasattr(predictor, 'predict'):
                        results = predictor.predict(image_rgb)
                    else:
                        # 备选：使用call接口
                        results = predictor(image_rgb)

                    if results is None:
                        self.log_signal.emit("SAM3推理返回None", "WARNING")
                        return np.array([]), np.array([]), np.array([])

                    # 处理结果
                    if isinstance(results, list) and len(results) > 0:
                        result = results[0]
                        masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') and getattr(result, 'masks', None) is not None else np.array([])
                    elif hasattr(results, 'masks'):
                        masks = results.masks.data.cpu().numpy() if getattr(results, 'masks', None) is not None else np.array([])
                    else:
                        masks = np.array([])

                    scores = []
                    logits = []

                    if len(masks) == 0:
                        self.log_signal.emit("SAM3推理返回空掩码", "WARNING")
                        return masks, np.array(scores), np.array(logits)

                    # 保留所有候选掩码；如果安装了CLIP再做语义排序
                    if text_prompt:
                        match_scores = self._clip_rerank_masks(image_rgb, masks, text_prompt)
                        if match_scores is not None and len(match_scores) > 0:
                            valid_idx = np.where(match_scores >= self.clip_threshold)[0]
                            if len(valid_idx) == 0:
                                self.log_signal.emit(
                                    f"无掩码达到相似度阈值 {self.clip_threshold:.2f}，返回原始候选结果", "WARNING"
                                )
                                valid_idx = np.arange(len(match_scores))

                            # 取相似度最好的 top_n 实例
                            top_idx = valid_idx[np.argsort(match_scores[valid_idx])[::-1][:self.top_n]]
                            masks = masks[top_idx]
                            scores = np.array(match_scores)[top_idx]
                            self.log_signal.emit(
                                f"根据文本匹配获得 {len(masks)} 个候选实例 (Top{self.top_n}, 阈值{self.clip_threshold:.2f})", "INFO"
                            )
                        else:
                            self.log_signal.emit(
                                "未检测到CLIP，使用全部SAM3推理候选", "INFO"
                            )
                            scores = np.array([1.0] * len(masks))
                            logits = np.array([0.0] * len(masks))
                    else:
                        scores = np.array([1.0] * len(masks))
                        logits = np.array([0.0] * len(masks))

                    return masks, scores, logits

                except Exception as e:
                    self.log_signal.emit(f"SAM3文本匹配异常: {e}，尝试简单推理", "WARNING")
                    # 降级处理：尝试简单推理
                    try:
                        if hasattr(predictor, 'predict'):
                            results = predictor.predict(image_rgb)
                        else:
                            results = predictor(image_rgb)

                        if isinstance(results, list) and len(results) > 0:
                            result = results[0]
                            masks = result.masks.data.cpu().numpy() if hasattr(result, 'masks') else np.array([])
                        elif hasattr(results, 'masks'):
                            masks = results.masks.data.cpu().numpy() if getattr(results, 'masks', None) is not None else np.array([])
                        else:
                            masks = np.array([])

                        return masks, np.array([1.0] * len(masks)), np.array([0.0] * len(masks))
                    except Exception as e2:
                        self.log_signal.emit(f"SAM3推理完全失败: {e2}", "ERROR")
                        return np.array([]), np.array([]), np.array([])
            else:
                # 不支持传统SAM模型
                self.log_signal.emit("不支持传统SAM模型，仅支持Ultralytics SAM3", "ERROR")
                return np.array([]), np.array([]), np.array([])

        except Exception as e:
            self.log_signal.emit(f"文字提示处理失败: {str(e)}", "WARNING")
            return np.array([]), np.array([]), np.array([])

    def _clip_rerank_masks(self, image_rgb, masks, text_prompt):
        """使用CLIP基于文本对候选掩码重排，支持中文语义匹配。"""
        try:
            import numpy as np
            import torch
            try:
                import open_clip
            except ImportError:
                try:
                    from transformers import CLIPProcessor, CLIPModel
                except ImportError:
                    self.log_signal.emit("未安装open_clip或transformers，无法进行CLIP文本匹配", "WARNING")
                    return None

            # 生成文本特征
            use_open_clip = False
            try:
                import open_clip as _open_clip
                use_open_clip = True
            except ImportError:
                _open_clip = None

            if use_open_clip:
                model_name = 'ViT-H-14'
                model, _, preprocess = _open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b79k')
                tokenizer = _open_clip.tokenize
                model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                text_tokens = tokenizer([text_prompt]).to(device)
                with torch.no_grad():
                    text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            else:
                processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
                model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
                model.eval()
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = model.to(device)
                text_inputs = processor(text=text_prompt, return_tensors='pt', padding=True)
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                with torch.no_grad():
                    text_features = model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            scores = []
            # 对每个掩码生成候选图像特征
            for mask in masks:
                if mask.ndim == 3:
                    mask = mask[0]
                mask_bool = mask > 0
                if not mask_bool.any():
                    scores.append(0.0)
                    continue

                # 抠出掩码所在区域
                import cv2
                from PIL import Image
                mask_img = np.zeros_like(image_rgb)
                mask_img[mask_bool] = image_rgb[mask_bool]
                nonzero = np.where(mask_bool)
                y1, y2 = nonzero[0].min(), nonzero[0].max()
                x1, x2 = nonzero[1].min(), nonzero[1].max()
                crop = mask_img[y1:y2+1, x1:x2+1]
                if crop.size == 0:
                    scores.append(0.0)
                    continue

                pil = Image.fromarray(crop.astype(np.uint8))

                if use_open_clip:
                    image_input = preprocess(pil).unsqueeze(0).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                else:
                    image_input = processor(images=pil, return_tensors='pt').to(device)
                    with torch.no_grad():
                        image_features = model.get_image_features(**image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                sim = (image_features @ text_features.T).item()
                scores.append(float(sim))

            return np.array(scores)

        except Exception as e:
            self.log_signal.emit(f"CLIP匹配异常，使用全部候选: {e}", "WARNING")
            return None

    def _masks_to_boxes(self, masks):
        """将掩码转换为方框列表"""
        import numpy as np

        if masks is None:
            return np.array([])

        if len(masks) == 0:
            return np.array([])

        boxes = []
        for mask in masks:
            if hasattr(mask, 'cpu'):
                mask = mask.cpu().numpy()
            if mask.ndim == 3:
                mask = mask[0]
            mask_bool = mask > 0
            if mask_bool.any():
                rows = np.any(mask_bool, axis=1)
                cols = np.any(mask_bool, axis=0)
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                boxes.append([xmin, ymin, xmax, ymax])
            else:
                boxes.append([0, 0, 0, 0])

        return np.array(boxes)

    def _infer_ultralytics_sam3(self, image_rgb, text_prompt):
        """Ultralytics SAM3推理（支持文本匹配降级方案）。"""
        import numpy as np
        try:
            if text_prompt:
                masks, scores, logits = self._infer_with_text_prompt(
                    self.model, image_rgb, text_prompt, ultralytics=True
                )
            else:
                results = self.model(image_rgb, verbose=False)
                if len(results) > 0:
                    result = results[0]
                    masks = result.masks.data.cpu().numpy() if getattr(result, 'masks', None) is not None else np.array([])
                    scores = []
                    logits = []
                else:
                    masks, scores, logits = np.array([]), np.array([]), np.array([])

            boxes = self._masks_to_boxes(masks)
            return masks, scores, logits, boxes

        except Exception as e:
            self.log_signal.emit(f"Ultralytics SAM3推理失败: {e}", "ERROR")
            return np.array([]), np.array([]), np.array([]), np.array([])

    def stop(self):
        """停止推理线程"""
        self.is_running = False

