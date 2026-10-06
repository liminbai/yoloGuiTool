import os
import traceback

import ultralytics
import yaml
from ultralytics import YOLO

from PySide6.QtCore import QThread, Signal


# ============================================
# 命名 / 参数常量（与 ultralytics 8.4 官方资产名保持一致）
# ============================================
# GUI 与配置里使用 yolov11*/yolov26*，但官方权重与模型 YAML 叫 yolo11*/yolo26*
# （实测：YOLO("yolov11n.pt") / YOLO("yolov11m-seg.yaml") 都会 FileNotFoundError）
MODEL_NAME_ALIASES = (("yolov11", "yolo11"), ("yolov26", "yolo26"))

# 各系列兜底权重（配置里的模型名非法/与所选系列不符时使用）
FAMILY_FALLBACK_WEIGHT = {
    "yolov8": "yolov8n.pt",
    "yolo11": "yolo11n.pt",
    "yolo26": "yolo26n.pt",
}

# 检查点保存周期（回调里判断“已保存检查点”必须用同一个值）
SAVE_PERIOD = 10

# DataLoader worker 数：Windows 下 DataLoader 走 spawn，会在 GUI 进程里重复导入
# PySide6/torch 且容易出现难排查的卡顿，因此默认 0（主进程加载）。
# 需要在 Linux 或追求吞吐时可显式设置 config["training"]["workers"]。
DEFAULT_WORKERS = 0 if os.name == "nt" else 8

# “数据增强”开关对应的训练增强参数（ultralytics 默认值）
AUGMENT_ON = {
    "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
    "degrees": 0.0, "translate": 0.1, "scale": 0.5,
    "shear": 0.0, "perspective": 0.0,
    "flipud": 0.0, "fliplr": 0.5,
    "mosaic": 1.0, "mixup": 0.0, "copy_paste": 0.0,
}
AUGMENT_OFF = {key: 0.0 for key in AUGMENT_ON}

# mAP 取值优先级（先分割，再检测，最后分类）
MAP_KEY_PRIORITY = (
    "metrics/mAP50-95(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(B)",
    "metrics/mAP50(B)",
    "metrics/accuracy_top1",
)


def normalize_family(family):
    """把 "YOLOv8"/"yolov11"/"yolo26" 之类的写法统一成 yolov8 / yolo11 / yolo26。

    注意：历史代码里 __init__ 用 "yolov8"/"yolov11" 判断，而
    prepare_training_args 用 "yolo11" 判断，导致 YOLO11 的专属参数从来没生效过。
    """
    text = str(family or "").strip().lower()
    if "8" in text:
        return "yolov8"
    if "11" in text:
        return "yolo11"
    if "26" in text:
        return "yolo26"
    return text


def asset_model_type(model_type, family):
    """GUI 模型名 -> 官方资产名（不含扩展名），例如 yolov11m-seg -> yolo11m-seg。"""
    name = str(model_type or "").strip().lower()
    for src, dst in MODEL_NAME_ALIASES:
        if name.startswith(src):
            name = dst + name[len(src):]
            break

    if not name.startswith(family):  # 与所选系列不一致时退回该系列默认模型
        return FAMILY_FALLBACK_WEIGHT.get(family, "yolov8n.pt").removesuffix(".pt")

    return name


def weight_filename(model_type, family):
    """返回预训练权重文件名，例如 yolov11m-seg.pt -> yolo11m-seg.pt。"""
    return f"{asset_model_type(model_type, family)}.pt"


def config_filename(model_type, family):
    """返回“从零构建”使用的模型 YAML 名，例如 yolov11m-seg -> yolo11m-seg.yaml。

    实测 ultralytics 支持带尺寸字母的写法（yolo11m-seg.yaml / yolo26m-seg.yaml 均可正常解析）。
    """
    return f"{asset_model_type(model_type, family)}.yaml"


def task_of_model_type(model_type):
    """从模型名推断任务类型，例如 yolov8m-seg -> segment。"""
    name = str(model_type or "").lower()
    if "-seg" in name:
        return "segment"
    if "-cls" in name:
        return "classify"
    return "detect"


def pick_map_score(metrics):
    """从 trainer.metrics 中挑一个用于展示的 mAP/accuracy；取不到返回 None。"""
    if not metrics:
        return None
    for key in MAP_KEY_PRIORITY:
        if key in metrics:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    return None


def augment_params(enabled):
    """返回训练增强参数字典（勾选=ultralytics 默认增强，未勾选=全部关闭）。"""
    return dict(AUGMENT_ON if enabled else AUGMENT_OFF)


def cuda_available():
    """GPU 是否可用（用于选择 device）。"""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def write_data_yaml(config, log=None):
    """按 GUI 配置写出 data_config.yaml，返回文件路径。

    只在 train 为相对路径时写 ``path`` 键：绝对路径下写它容易让人误以为
    它是数据集根目录，而且 ultralytics 本来就会以 YAML 所在目录为基准解析相对路径。
    """
    class_names = config["dataset"]["names"]
    names_dict = {i: name for i, name in enumerate(class_names)}

    train_path = config["dataset"]["train"]
    data_config = {
        "train": train_path,
        "val": config["dataset"]["val"],
        "test": config["dataset"]["test"] or None,
        "nc": len(class_names),
        "names": names_dict,
    }
    if train_path and not os.path.isabs(train_path):
        data_config["path"] = os.path.dirname(train_path) or "."

    save_dir = config["model"]["save_dir"] or "."
    os.makedirs(save_dir, exist_ok=True)

    data_yaml_path = os.path.join(save_dir, "data_config.yaml")
    data_config = {k: v for k, v in data_config.items() if v is not None}
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    if log:
        log(f"数据配置文件已创建: {data_yaml_path}", "INFO")
        log(f"类别配置(带序号): {names_dict}", "INFO")

    return data_yaml_path


def find_best_weights(config, extra_candidates=()):
    """定位可用权重：先看传入的候选（如刚训练完的 trainer.best/last），
    再按 ``<save_dir>/<weight_name>/weights/{best,last}.pt`` 与
    ``<save_dir>/weights/{best,last}.pt`` 查找；均不存在时返回 None。"""
    candidates = [path for path in extra_candidates if path]
    save_dir = config["model"].get("save_dir") or "."
    weight_name = config["model"].get("weight_name") or ""

    run_dirs = [os.path.join(save_dir, weight_name)] if weight_name else []
    run_dirs.append(save_dir)
    for run_dir in run_dirs:
        candidates.append(os.path.join(run_dir, "weights", "best.pt"))
        candidates.append(os.path.join(run_dir, "weights", "last.pt"))

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


# ============================================
# YOLO训练线程（支持YOLOv8、YOLO11和YOLOv26）
# ============================================
class YOLOTrainingThread(QThread):
    """YOLO训练线程 - 支持YOLOv8、YOLOv11和YOLOv26"""
    
    # 定义信号
    log_signal = Signal(str, str)  # 日志信号 (消息, 级别)
    progress_signal = Signal(int, int, float, float, float, float)  # 进度信号
    training_complete_signal = Signal(bool, str)  # 训练完成信号
    checkpoint_saved_signal = Signal(str)  # 检查点保存信号
    epoch_start_signal = Signal(int, int)  # 轮次开始信号
    epoch_end_signal = Signal(int, int, float, float, float, float)  # 轮次结束信号
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True
        self.stop_requested = False      # GUI 点“停止”后置位，用于收尾时给出正确提示
        self.model = None
        self.current_epoch = 0
        self.total_epochs = config["training"]["epochs"]
        self.model_type = config["model"]["type"]
        self.model_family = normalize_family(config["model"]["family"])  # yolov8 / yolo11 / yolo26
        self.weight_file = weight_filename(self.model_type, self.model_family)
        # 以模型名推断任务（与 ultralytics 内部 _smart_load 的依据一致），
        # 传入的 task 只用于校验与提示，不再塞进 train() 的 overrides
        self.task = task_of_model_type(self.model_type)
        self.configured_task = str(config["model"].get("task") or self.task).lower()
        self.last_map_score = 0.0        # 最近一次验证得到的真实 mAP（供进度显示）
        self.last_loss = 0.0             # 最近一轮的 loss / lr，训练结束时回填到最终进度
        self.last_lr = 0.0
        
    def run(self):
        """执行训练 - 使用YOLO Python API"""
        try:
            family_label = {
                "yolov8": "YOLOv8", "yolo11": "YOLO11", "yolo26": "YOLO26"
            }.get(self.model_family, self.model_family.upper())
            task_label = {
                "detect": "检测", "segment": "分割", "classify": "分类"
            }.get(self.task, self.task)
            self.log_signal.emit(
                f"开始 {family_label} {task_label}训练 (模型: {self.weight_file})...", "INFO"
            )

            # ultralytics 以“模型”决定实际任务，配置里的 task 若不一致必须提示
            if self.configured_task != self.task:
                self.log_signal.emit(
                    f"⚠️ 任务类型与模型不匹配：配置为 {self.configured_task}，"
                    f"但模型 {self.model_type} 属于 {self.task} 任务。"
                    f"ultralytics 将以模型任务 {self.task} 为准，请检查模型/任务选择。",
                    "WARNING",
                )
            
            self.log_signal.emit(f"使用ultralytics版本: {ultralytics.__version__}", "INFO")
            
            # 1. 准备训练参数
            train_args = self.prepare_training_args()
            
            # 2. 加载模型
            model_file = self.get_model_file()
            self.log_signal.emit(f"加载模型: {model_file}", "INFO")
            
            try:
                # 与 CLI（yolo segment train model=xxx.pt）保持一致：任务交给构造函数，
                # 而不是塞进 train() 的 overrides（实测传进去也会被模型自身任务覆盖）
                if self.config["model"]["pretrained"]:
                    self.model = YOLO(model_file, task=self.task)
                else:
                    yaml_file = config_filename(self.model_type, self.model_family)
                    self.model = YOLO(yaml_file, task=self.task)
                    self.log_signal.emit(f"已从配置文件创建新模型: {yaml_file}", "INFO")
            except Exception as e:
                fallback = FAMILY_FALLBACK_WEIGHT.get(self.model_family, "yolov8n.pt")
                self.log_signal.emit(
                    f"加载模型失败({model_file})，回退到 {fallback}: {str(e)}", "WARNING"
                )
                try:
                    self.model = YOLO(fallback, task=self.task)
                    self.log_signal.emit(
                        f"⚠️ 已改用 {fallback} 继续训练，这与你选择的模型不一致！", "WARNING"
                    )
                except Exception as e2:
                    error_msg = f"加载模型完全失败: {str(e2)}"
                    self.log_signal.emit(error_msg, "ERROR")
                    self.training_complete_signal.emit(False, error_msg)
                    return
            
            # 3. 添加训练回调
            self.add_training_callbacks()
            
            # 4. 执行训练
            self.log_signal.emit("开始训练过程...", "INFO")
            # 注意：train() 返回的是 dict 型指标（detect/segment 为 {Task}Metrics），
            # 没有 .best/.metrics 属性，最佳权重要从 trainer 上取
            self.model.train(**train_args)

            # 5. 训练完成：权重路径与真实指标均取自 trainer
            trainer = getattr(self.model, "trainer", None)
            best_weights = getattr(trainer, "best", None)
            save_dir = getattr(trainer, "save_dir", None)
            metrics = dict(getattr(trainer, "metrics", None) or {})
            final_map = pick_map_score(metrics)

            if best_weights and os.path.isfile(best_weights):
                self.log_signal.emit(f"最佳权重: {best_weights}", "SUCCESS")
            if save_dir:
                self.log_signal.emit(f"训练结果目录: {save_dir}", "INFO")
            if metrics:
                self.log_signal.emit(f"最终指标: {metrics}", "INFO")
            else:
                self.log_signal.emit("未取到验证指标（val 未执行或训练被提前停止）", "WARNING")

            # 发送最终进度（使用真实指标，不再写死 0.85）
            self.epoch_end_signal.emit(
                self.current_epoch or self.total_epochs,
                self.total_epochs,
                float(self.last_loss),
                float(self.last_lr),
                float(final_map if final_map is not None else self.last_map_score),
                100.0,
            )

            if self.stop_requested:
                result_info = "训练已被用户停止"
                if best_weights and os.path.isfile(best_weights):
                    result_info += f"，已保存权重: {best_weights}"
                self.log_signal.emit(result_info, "WARNING")
            else:
                result_info = f"{self.model_family.upper()} 训练成功完成"
                if best_weights and os.path.isfile(best_weights):
                    result_info += f"，最佳权重: {best_weights}"
                if final_map is not None:
                    result_info += f"，mAP: {final_map:.4f}"
                self.log_signal.emit(f"{self.model_family.upper()} 训练完成！", "SUCCESS")

            self.training_complete_signal.emit(True, result_info)

        except Exception as e:
            error_msg = f"训练出错: {str(e)}"
            self.log_signal.emit(error_msg, "ERROR")
            self.log_signal.emit(traceback.format_exc(), "ERROR")
            self.training_complete_signal.emit(False, error_msg)
    
    def get_model_file(self):
        """返回预训练权重文件名（已在 __init__ 里做过官方命名归一化）。"""
        return self.weight_file
    
    def add_training_callbacks(self):
        """使用新的API添加回调函数"""
        
        def on_train_start(trainer):
            """训练开始时调用"""
            self.log_signal.emit("训练开始...", "INFO")
            self.epoch_start_signal.emit(0, self.total_epochs)
        
        def on_train_epoch_start(trainer):
            """每个训练轮次开始时调用"""
            self.current_epoch = trainer.epoch + 1
            self.epoch_start_signal.emit(self.current_epoch, self.total_epochs)
            self.log_signal.emit(f"开始第 {self.current_epoch}/{self.total_epochs} 轮训练", "INFO")
        
        def on_train_epoch_end(trainer):
            """每个训练轮次结束时调用"""
            try:
                current_epoch = trainer.epoch + 1
                
                # 获取损失值
                loss = 0.0
                if hasattr(trainer, 'loss'):
                    if isinstance(trainer.loss, (int, float)):
                        loss = trainer.loss
                    elif hasattr(trainer.loss, 'item'):
                        loss = trainer.loss.item()
                    else:
                        if hasattr(trainer, 'loss_dict') and trainer.loss_dict:
                            for key, value in trainer.loss_dict.items():
                                if 'loss' in key.lower():
                                    if hasattr(value, 'item'):
                                        loss = value.item()
                                    elif isinstance(value, (int, float)):
                                        loss = value
                                    break
                
                # 获取学习率
                lr = 0.001
                if hasattr(trainer, 'lr'):
                    if isinstance(trainer.lr, (int, float)):
                        lr = trainer.lr
                    elif isinstance(trainer.lr, list) and len(trainer.lr) > 0:
                        lr = trainer.lr[0]
                
                # 计算进度
                progress = (current_epoch / self.total_epochs) * 100
                
                # mAP 用最近一次验证的真实值（val 在 on_fit_epoch_end 才更新），不再模拟增长
                map_score = self.last_map_score
                self.last_loss, self.last_lr = loss, lr
                
                # 发送进度信号
                self.epoch_end_signal.emit(
                    current_epoch, 
                    self.total_epochs, 
                    loss, 
                    lr, 
                    map_score, 
                    progress
                )
                
                # 每5个epoch记录一次详细信息
                if current_epoch % 5 == 0:
                    self.log_signal.emit(
                        f"Epoch {current_epoch}/{self.total_epochs} 完成, "
                        f"损失: {loss:.4f}, LR: {lr:.6f}, mAP: {map_score:.4f}", 
                        "INFO"
                    )
                
                # 检查点保存（与 save_period 保持同一常量）
                if SAVE_PERIOD > 0 and current_epoch % SAVE_PERIOD == 0:
                    self.checkpoint_saved_signal.emit(f"epoch_{current_epoch}")
                    
            except Exception as e:
                self.log_signal.emit(f"处理训练进度时出错: {str(e)}", "ERROR")
        
        def on_fit_epoch_end(trainer):
            """每个 epoch 的验证结束后调用：此时 trainer.metrics 已是真实指标。"""
            try:
                metrics = dict(getattr(trainer, 'metrics', None) or {})
                map_score = pick_map_score(metrics)
                if map_score is None:
                    return

                self.last_map_score = map_score
                summary = {
                    key: round(float(value), 4)
                    for key, value in metrics.items()
                    if 'mAP' in key or 'precision' in key or 'recall' in key
                }
                self.log_signal.emit(
                    f"Epoch {trainer.epoch + 1} 验证指标: {summary}", "INFO"
                )
            except Exception as e:
                self.log_signal.emit(f"读取验证指标时出错: {str(e)}", "ERROR")

        # 使用新的方法添加回调
        self.model.add_callback("on_train_start", on_train_start)
        self.model.add_callback("on_train_epoch_start", on_train_epoch_start)
        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
        self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    
    def prepare_training_args(self):
        """准备训练参数字典"""
        train_args = {}
        
        # 必需参数：数据配置文件路径
        if self.config["dataset"]["train"]:
            # 创建数据配置文件
            data_yaml_path = self.create_data_yaml()
            train_args['data'] = data_yaml_path
        
        # 关键训练参数
        train_args['epochs'] = self.config["training"]["epochs"]
        train_args['batch'] = self.config["training"]["batch_size"]
        train_args['imgsz'] = self.config["model"]["input_size"]
        train_args['lr0'] = self.config["training"]["lr"]
        
        # 优化器相关参数
        optimizer = self.config["training"]["optimizer"]
        train_args['optimizer'] = optimizer
        
        if optimizer == "SGD":
            train_args['momentum'] = self.config["training"]["momentum"]
        
        train_args['weight_decay'] = self.config["training"]["weight_decay"]
        train_args['warmup_epochs'] = self.config["training"]["warmup_epochs"]
        train_args['warmup_momentum'] = 0.8
        train_args['warmup_bias_lr'] = 0.1
        
        # 数据增强：ultralytics 的 `augment` 是“验证时 TTA”，不是训练增强，
        # 训练增强必须显式传 hsv_*/degrees/translate/scale/fliplr/mosaic 等参数
        augmentation = bool(self.config["training"]["augmentation"])
        train_args.update(augment_params(augmentation))
        train_args['close_mosaic'] = (
            self.config["training"].get("close_mosaic", 10) if augmentation else 0
        )
        
        # 早停机制
        if self.config["training"]["early_stopping"]:
            train_args['patience'] = self.config["training"]["patience"]
        
        # 保存路径和名称
        save_dir = self.config["model"]["save_dir"]
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            train_args['project'] = save_dir
        
        weight_name = self.config["model"]["weight_name"]
        if weight_name:
            train_args['name'] = weight_name
        
        # 任务类型不在此处传递：ultralytics 以模型自身决定任务
        # （实测：给 -seg 模型传 task='detect' 也会被覆盖），已在 YOLO(model, task=...) 指定
        
        # 其他有用的参数
        train_args['exist_ok'] = True
        train_args['save_period'] = SAVE_PERIOD
        train_args['workers'] = self.config["training"].get("workers", DEFAULT_WORKERS)
        train_args['device'] = '0' if self.check_gpu() else 'cpu'
        train_args['verbose'] = False
        train_args['deterministic'] = True
        
        # 各系列通用参数（cos_lr / label_smoothing 并非 v8 专有；
        # 分割相关参数对检测/分类任务是 no-op）
        train_args['cos_lr'] = self.config["training"].get("cos_lr", True)
        train_args['label_smoothing'] = self.config["training"].get("label_smoothing", 0.0)
        train_args['overlap_mask'] = self.config["training"].get("overlap_mask", True)
        train_args['mask_ratio'] = self.config["training"].get("mask_ratio", 4)
        # mixup / copy_paste：只在配置里显式给出时覆盖增强设定
        if "mixup" in self.config["training"]:
            train_args['mixup'] = self.config["training"]["mixup"]
        if "copy_paste" in self.config["training"]:
            train_args['copy_paste'] = self.config["training"]["copy_paste"]
        
        self.log_signal.emit(f"训练参数: {str(train_args)}", "INFO")
        return train_args
    
    def create_data_yaml(self):
        """创建数据配置文件 - 类别带序号（实现见模块级 write_data_yaml）"""
        return write_data_yaml(self.config, self.log_signal.emit)
    
    def check_gpu(self):
        """检查GPU是否可用"""
        return cuda_available()
    
    def stop(self):
        """请求停止训练。

        实测（ultralytics 8.4）：训练循环会在批次边界检查 trainer.stop，
        置位后 1 个 epoch 内即可退出，并正常保存 last.pt / best.pt。
        原先只置 is_running 标志，训练进程完全感知不到（会一直跑到最后）。
        """
        self.is_running = False
        self.stop_requested = True

        trainer = getattr(self.model, "trainer", None)
        if trainer is not None and hasattr(trainer, "stop"):
            trainer.stop = True
            self.log_signal.emit(
                "已请求停止训练：将在当前批次结束后退出并保存权重", "WARNING"
            )
        else:
            self.log_signal.emit("训练尚未开始或已结束，已取消后续操作", "WARNING")


# ============================================
# YOLO 评估线程（真实指标，供 GUI 的“快速测试”使用）
# ============================================
class YOLOValidationThread(QThread):
    """在 test（缺省 val）划分上评估已有权重。

    原先 GUI 的“快速测试”是用 random 生成精度/召回/mAP 的假结果，
    这里改为真正调用 ``model.val()``，指标全部来自验证器。
    """

    log_signal = Signal(str, str)
    # (是否成功, {weights, data_yaml, split, metrics, speed} | {error})
    val_complete_signal = Signal(bool, dict)

    def __init__(self, weights, config, split="test"):
        super().__init__()
        self.weights = weights
        self.config = config
        self.split = split
        self.results = {}

    def run(self):
        """执行评估，指标全部来自 ultralytics 验证器"""
        try:
            if not self.weights or not os.path.isfile(self.weights):
                raise FileNotFoundError(f"权重文件不存在: {self.weights}")

            split_name, hint = self.resolve_split()

            self.log_signal.emit(f"加载权重: {self.weights}", "INFO")
            model = YOLO(self.weights, task=task_of_model_type(self.config["model"]["type"]))

            data_yaml = write_data_yaml(self.config, self.log_signal.emit)

            if hint:
                self.log_signal.emit(hint, "WARNING")
            self.log_signal.emit(f"开始在 {split_name} 集上评估...", "INFO")

            metrics = model.val(
                data=data_yaml,
                split=split_name,
                imgsz=self.config["model"]["input_size"],
                batch=self.config["training"]["batch_size"],
                device='0' if cuda_available() else 'cpu',
                project=self.config["model"]["save_dir"] or ".",
                name="val_" + (self.config["model"]["weight_name"] or "run"),
                exist_ok=True,
                verbose=False,
            )

            results_dict = dict(getattr(metrics, "results_dict", None) or {})
            if not results_dict:
                raise RuntimeError("未取得评估指标（数据集为空或缺标注？）")

            self.results = {
                "weights": self.weights,
                "data_yaml": data_yaml,
                "split": split_name,
                "metrics": {key: float(value) for key, value in results_dict.items()},
                "speed": dict(getattr(metrics, "speed", None) or {}),
            }
            summary = ", ".join(f"{key}={float(value):.4f}" for key, value in results_dict.items())
            self.log_signal.emit(f"评估完成（{split_name}）: {summary}", "SUCCESS")
            self.val_complete_signal.emit(True, self.results)

        except Exception as e:
            self.log_signal.emit(f"评估失败: {str(e)}", "ERROR")
            self.log_signal.emit(traceback.format_exc(), "ERROR")
            self.val_complete_signal.emit(False, {"error": str(e)})

    def resolve_split(self):
        """决定在哪个划分上评估：没有可用的 test 集时退回 val。

        返回 (split_name, 提示信息)；数据集不可用时抛 FileNotFoundError。
        """
        dataset = self.config["dataset"]
        if self.split != "test":
            return self.split, ""

        test_path = dataset.get("test")
        if test_path and os.path.exists(test_path):
            return "test", ""

        val_path = dataset.get("val")
        if val_path and os.path.exists(val_path):
            return "val", "未配置有效的测试集，已改用验证集评估"

        raise FileNotFoundError("测试集与验证集都未配置或不存在，无法评估")


# ============================================
# 类别编辑器对话框
# ============================================
