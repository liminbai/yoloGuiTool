import json
import yaml

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QHBoxLayout, QGroupBox, QTextEdit,
    QPushButton, QDialogButtonBox, QFileDialog, QMessageBox
)


class ClassEditorDialog(QDialog):
    """类别编辑器对话框 - 支持带序号显示"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("编辑类别")
        self.setModal(True)
        self.setMinimumSize(600, 500)
        
        self.init_ui()
        
    def init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout()
        
        info_label = QLabel("每行输入一个类别名称（将自动添加序号）:")
        layout.addWidget(info_label)
        
        main_content_layout = QHBoxLayout()
        
        # 左侧：带序号的类别预览
        preview_group = QGroupBox("类别预览（带序号）")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        self.preview_text.setMaximumWidth(250)
        self.preview_text.setStyleSheet("background-color: #f8f9fa;")
        preview_layout.addWidget(self.preview_text)
        
        preview_group.setLayout(preview_layout)
        main_content_layout.addWidget(preview_group)
        
        # 右侧：类别编辑区域
        edit_group = QGroupBox("类别编辑")
        edit_layout = QVBoxLayout()
        
        self.classes_edit = QTextEdit()
        self.classes_edit.setPlaceholderText("例如:\nperson\ncar\ndog\ncat\n...")
        self.classes_edit.textChanged.connect(self.update_preview)
        edit_layout.addWidget(self.classes_edit)
        
        edit_group.setLayout(edit_layout)
        main_content_layout.addWidget(edit_group)
        
        layout.addLayout(main_content_layout)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("从文件加载")
        self.load_btn.clicked.connect(self.load_from_file)
        button_layout.addWidget(self.load_btn)
        
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self.clear_classes)
        button_layout.addWidget(self.clear_btn)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 类别统计
        self.count_label = QLabel("类别数量: 0")
        layout.addWidget(self.count_label)
        
        # 对话框按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
    def update_preview(self):
        """更新带序号的预览"""
        classes = self.get_classes()
        preview_lines = []
        
        for i, class_name in enumerate(classes):
            preview_lines.append(f"{i}: {class_name}")
        
        self.preview_text.setPlainText("\n".join(preview_lines))
        self.count_label.setText(f"类别数量: {len(classes)}")
        
    def set_classes(self, classes):
        """设置类别"""
        text = "\n".join(classes)
        self.classes_edit.setPlainText(text)
        self.update_preview()
        
    def get_classes(self):
        """获取类别列表"""
        text = self.classes_edit.toPlainText().strip()
        if not text:
            return []
        
        classes = [line.strip() for line in text.split('\n') if line.strip()]
        return classes
    
    def load_from_file(self):
        """从文件加载类别"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择类别文件", "", 
            "文本文件 (*.txt);;JSON文件 (*.json);;YAML文件 (*.yaml *.yml);;所有文件 (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                classes = []
                
                if file_path.endswith('.json'):
                    data = json.loads(content)
                    if isinstance(data, list):
                        classes = [str(item) for item in data]
                    elif isinstance(data, dict):
                        if 'names' in data:
                            names_data = data['names']
                            if isinstance(names_data, dict):
                                classes = [names_data[str(i)] for i in sorted(map(int, names_data.keys()))]
                            elif isinstance(names_data, list):
                                classes = [str(item) for item in names_data]
                        else:
                            try:
                                sorted_items = sorted(data.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
                                classes = [str(value) for key, value in sorted_items]
                            except:
                                classes = list(data.values())
                elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    data = yaml.safe_load(content)
                    if isinstance(data, list):
                        classes = [str(item) for item in data]
                    elif isinstance(data, dict):
                        if 'names' in data:
                            names_data = data['names']
                            if isinstance(names_data, dict):
                                classes = [names_data[str(i)] for i in sorted(map(int, names_data.keys()))]
                            elif isinstance(names_data, list):
                                classes = [str(item) for item in names_data]
                        else:
                            try:
                                sorted_items = sorted(data.items(), key=lambda x: int(x[0]) if isinstance(x[0], (int, str)) and str(x[0]).isdigit() else x[0])
                                classes = [str(value) for key, value in sorted_items]
                            except:
                                classes = list(data.values())
                else:
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line:
                            if ': ' in line:
                                parts = line.split(': ', 1)
                                if len(parts) == 2 and parts[0].strip().isdigit():
                                    classes.append(parts[1].strip())
                                else:
                                    classes.append(line)
                            else:
                                classes.append(line)
                
                self.set_classes(classes)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文件失败:\n{str(e)}")
                
    def clear_classes(self):
        """清空类别"""
        self.classes_edit.clear()

