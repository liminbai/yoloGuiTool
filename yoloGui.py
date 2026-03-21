"""兼容入口：保留历史启动方式 `python yoloGui.py`。"""

from gui.windows.trainer_main_window import YOLOTrainerGUI, main

__all__ = ["YOLOTrainerGUI", "main"]


if __name__ == "__main__":
    main()
