# gui/main.py

import sys
import os
import logging
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QCheckBox, QProgressBar, QMessageBox, QTabWidget,
    QScrollArea, QSlider, QTextEdit, QListWidget, QListWidgetItem, QAbstractItemView, QSizePolicy, QSplitter, QGroupBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QObject
from PyQt6.QtGui import QPixmap, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

from hdr_analysis.analyzer import HDRAnalyzer

# 自定义日志处理器，将日志信息发送到GUI
class QtHandler(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self)
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

# QThread用于执行分析任务，避免阻塞主线程
class AnalyzerThread(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, analyzer):
        super().__init__()
        self.analyzer = analyzer

    def run(self):
        try:
            self.analyzer.analyze()
            # 读取分析报告
            if not self.analyzer.only_3d:
                results = self.load_results()
            else:
                results = []
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

    def load_results(self):
        # 读取分析报告CSV
        if os.path.exists(self.analyzer.report_path):
            df = pd.read_csv(self.analyzer.report_path)
            return df.to_dict(orient='records')
        else:
            return []

# 用于显示单个图像及其缩放滑块的自定义小部件
class ImageDisplay(QWidget):
    def __init__(self, title, image_path):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 标题
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.title_label)

        # 图像显示
        self.pixmap = QPixmap(image_path).scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.image_label = QLabel()
        self.image_label.setPixmap(self.pixmap)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        # 缩放滑块
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(50, 200)  # 缩放范围 50% - 200%
        self.slider.setValue(100)
        self.slider.valueChanged.connect(self.scale_image)
        self.layout.addWidget(self.slider)

    def scale_image(self, value):
        scaled_pixmap = self.pixmap.scaled(
            self.pixmap.width() * value // 100,
            self.pixmap.height() * value // 100,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

# 用于显示每个数据集的分析结果（多张图像）
class ResultDisplay(QWidget):
    def __init__(self, analysis_dir, set_name):
        super().__init__()

        self.analysis_dir = analysis_dir
        self.set_name = set_name

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 标题
        self.title_label = QLabel(f"数据集: {set_name}")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.title_label)

        # 图像布局
        self.image_layout = QHBoxLayout()
        self.layout.addLayout(self.image_layout)

        # 添加各类图像
        self.add_image("差异图像", f"{set_name}_diff.png")
        self.add_image("3D亮度图", f"{set_name}_亮度3D图.png")
        self.add_image("功率谱", f"{set_name}_功率谱.png")
        self.add_image("频域熵", f"{set_name}_频域熵.png")

    def add_image(self, title, filename):
        image_path = os.path.join(self.analysis_dir, filename)
        if os.path.exists(image_path):
            image_display = ImageDisplay(title, image_path)
            self.image_layout.addWidget(image_display)

# 主窗口类，集成所有GUI组件
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDR 图像分析工具")
        self.setGeometry(100, 100, 1200, 800)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # 设置日志处理
        self.logger_handler = QtHandler()
        self.logger_handler.log_signal.connect(self.append_log)
        logging.getLogger().addHandler(self.logger_handler)
        logging.getLogger().setLevel(logging.INFO)

        # 创建GUI组件
        self.create_input_output_selection()
        self.create_image_selection()
        self.create_options_selection()
        self.create_start_button()
        self.create_progress_bar()
        self.create_log_viewer()
        self.create_tabs()

        self.analyzer = None
        self.thread = None

        # 设置拖放支持
        self.setAcceptDrops(True)

    def create_input_output_selection(self):
        layout = QHBoxLayout()

        # 输入目录选择
        self.input_label = QLabel("输入目录:")
        self.input_path = QLabel("未选择")
        self.input_path.setStyleSheet("border: 1px solid gray; padding: 2px;")
        self.input_button = QPushButton("选择")
        self.input_button.clicked.connect(self.select_input_dir)

        layout.addWidget(self.input_label)
        layout.addWidget(self.input_path)
        layout.addWidget(self.input_button)

        # 输出目录选择
        self.output_label = QLabel("输出目录:")
        self.output_path = QLabel("未选择")
        self.output_path.setStyleSheet("border: 1px solid gray; padding: 2px;")
        self.output_button = QPushButton("选择")
        self.output_button.clicked.connect(self.select_output_dir)

        layout.addWidget(self.output_label)
        layout.addWidget(self.output_path)
        layout.addWidget(self.output_button)

        self.layout.addLayout(layout)

    def create_image_selection(self):
        # 图像选择列表
        self.image_list = QListWidget()
        self.image_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.image_list.setIconSize(QSize(100, 100))
        self.image_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_list.setSpacing(10)
        self.image_list.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # 添加到布局中
        image_selection_layout = QVBoxLayout()
        image_selection_layout.addWidget(QLabel("选择要分析的照片:"))
        image_selection_layout.addWidget(self.image_list)

        # 使用 QSplitter 使图像选择部分可调整大小
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter_widget = QWidget()
        splitter_widget.setLayout(image_selection_layout)
        splitter.addWidget(splitter_widget)
        splitter.setStretchFactor(0, 1)

        self.layout.addWidget(splitter)

    def create_options_selection(self):
        layout = QHBoxLayout()

        # 图像类型选择
        self.image_type_label = QLabel("图像类型:")
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(['original', 'fused', 'both'])
        self.image_type_combo.currentIndexChanged.connect(self.update_image_selection_based_on_type)

        layout.addWidget(self.image_type_label)
        layout.addWidget(self.image_type_combo)

        # 日志等级选择
        self.log_level_label = QLabel("日志等级:")
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.log_level_combo.setCurrentText('INFO')  # 设置默认日志等级
        self.log_level_combo.currentTextChanged.connect(self.change_log_level)

        layout.addWidget(self.log_level_label)
        layout.addWidget(self.log_level_combo)

        # 分析选项选择
        self.analysis_group = QGroupBox("选择要执行的分析")
        analysis_layout = QHBoxLayout()

        self.stat_analysis_checkbox = QCheckBox("统计分析")
        self.freq_analysis_checkbox = QCheckBox("频域分析")
        self.diff_image_checkbox = QCheckBox("差异图像生成")
        self.brightness_3d_checkbox = QCheckBox("3D亮度图生成")

        # 默认选中统计分析和频域分析，取消3D亮度图生成
        self.stat_analysis_checkbox.setChecked(True)
        self.freq_analysis_checkbox.setChecked(True)
        self.diff_image_checkbox.setChecked(True)
        self.brightness_3d_checkbox.setChecked(False)  # 用户可以选择是否生成

        analysis_layout.addWidget(self.stat_analysis_checkbox)
        analysis_layout.addWidget(self.freq_analysis_checkbox)
        analysis_layout.addWidget(self.diff_image_checkbox)
        analysis_layout.addWidget(self.brightness_3d_checkbox)

        self.analysis_group.setLayout(analysis_layout)

        layout.addWidget(self.analysis_group)

        self.layout.addLayout(layout)

    def create_start_button(self):
        layout = QHBoxLayout()
        self.start_button = QPushButton("开始分析")
        self.start_button.clicked.connect(self.start_analysis)
        layout.addStretch()
        layout.addWidget(self.start_button)
        self.layout.addLayout(layout)

    def create_progress_bar(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layout.addWidget(self.progress_bar)

    def create_log_viewer(self):
        layout = QVBoxLayout()
        self.log_label = QLabel("日志输出:")
        layout.addWidget(self.log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_text)

        self.layout.addLayout(layout)

    def create_tabs(self):
        self.tabs = QTabWidget()
        self.image_tab = QWidget()
        self.stats_tab = QWidget()
        self.tabs.addTab(self.image_tab, "图像比较")
        self.tabs.addTab(self.stats_tab, "统计数据")

        # 图像比较布局
        self.image_layout = QVBoxLayout()
        self.image_scroll = QScrollArea()
        self.image_container = QWidget()
        self.image_layout_inner = QVBoxLayout()
        self.image_container.setLayout(self.image_layout_inner)
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_container)
        self.image_layout.addWidget(self.image_scroll)
        self.image_tab.setLayout(self.image_layout)

        # 统计数据布局
        self.stats_layout = QVBoxLayout()
        self.stats_canvas = FigureCanvas(plt.Figure())
        self.stats_layout.addWidget(self.stats_canvas)
        self.stats_tab.setLayout(self.stats_layout)

        self.layout.addWidget(self.tabs)

    def select_input_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输入目录")
        if dir_path:
            self.input_path.setText(dir_path)
            # 自动设置输出目录为 input_dir/output
            default_output = os.path.join(dir_path, "output")
            self.output_path.setText(default_output)
            # 自动加载图像
            self.load_images(dir_path)

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_path.setText(dir_path)

    def load_images(self, input_dir):
        # 清空当前列表
        self.image_list.clear()
        # 支持的图像格式
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        # 获取当前图像类型选择
        current_type = self.image_type_combo.currentText()

        # 根据图像类型过滤文件
        if current_type == 'original':
            # 选择不以 _fused 结尾的图像
            image_files = [f for f in os.listdir(input_dir)
                           if f.lower().endswith(supported_formats) and not (f.lower().endswith('_fused.jpg') or f.lower().endswith('_fused.png'))]
        elif current_type == 'fused':
            # 选择以 _fused 结尾的图像
            image_files = [f for f in os.listdir(input_dir)
                           if f.lower().endswith(supported_formats) and (f.lower().endswith('_fused.jpg') or f.lower().endswith('_fused.png'))]
        else:
            # both: 所有图像
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]

        # 遍历输入目录中的图像文件
        for filename in image_files:
            filepath = os.path.join(input_dir, filename)
            pixmap = QPixmap(filepath)
            if pixmap.isNull():
                continue  # 忽略无法加载的图像
            icon = QIcon(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            item = QListWidgetItem(icon, filename)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)  # 默认选中
            self.image_list.addItem(item)

    def update_image_selection_based_on_type(self):
        """
        根据图像类型选择，更新图像列表的选择状态。
        """
        if not self.input_path.text() or self.input_path.text() == "未选择":
            return

        input_dir = self.input_path.text()
        current_type = self.image_type_combo.currentText()

        # 重新加载图像列表
        self.load_images(input_dir)

    def change_log_level(self, level):
        """
        根据用户选择的日志等级，设置日志的详细程度。
        """
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            return
        logging.getLogger().setLevel(numeric_level)
        self.logger_handler.setLevel(numeric_level)

    def start_analysis(self):
        input_dir = self.input_path.text()
        output_dir = self.output_path.text()

        # 获取选中的图像文件
        selected_images = []
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                selected_images.append(os.path.join(input_dir, item.text()))

        # 获取用户选择的分析选项
        selected_analyses = {
            'statistical': self.stat_analysis_checkbox.isChecked(),
            'frequency': self.freq_analysis_checkbox.isChecked(),
            'difference': self.diff_image_checkbox.isChecked(),
            'brightness_3d': self.brightness_3d_checkbox.isChecked()
        }

        # 检查至少选择一个分析选项
        if not any(selected_analyses.values()):
            QMessageBox.warning(self, "警告", "请至少选择一个分析选项。")
            return

        if not selected_images:
            QMessageBox.warning(self, "警告", "请至少选择一张要分析的照片。")
            return

        # 设置默认输出目录为 input_dir/output
        if self.output_path.text() == "未选择":
            output_dir = os.path.join(input_dir, "output")
            self.output_path.setText(output_dir)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 初始化HDRAnalyzer
        self.analyzer = HDRAnalyzer(
            selected_images=selected_images,
            output_dir=output_dir,
            report_path=os.path.join(output_dir, 'analysis_report.csv'),
            analysis_dir=os.path.join(output_dir, 'analysis_results'),
            image_type=self.image_type_combo.currentText(),
            analyses=selected_analyses
        )

        # 启动分析线程
        self.thread = AnalyzerThread(self.analyzer)
        self.thread.finished.connect(self.analysis_finished)
        self.thread.error.connect(self.analysis_error)
        self.thread.start()

        # 显示进度条为无限循环
        self.progress_bar.setRange(0, 0)
        self.start_button.setEnabled(False)

    def analysis_finished(self, results):
        # 恢复进度条
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "完成", "HDR图像分析已完成。")
        self.start_button.setEnabled(True)
        self.display_results(results)

    def analysis_error(self, error_msg):
        # 恢复进度条
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "错误", f"分析过程中出现错误:\n{error_msg}")
        self.start_button.setEnabled(True)

    def display_results(self, results):
        # 清空之前的图像
        while self.image_layout_inner.count():
            child = self.image_layout_inner.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # 显示每个结果集的图像
        for result in results:
            set_name = result.get('Set', 'Unknown')
            result_display = ResultDisplay(self.analyzer.analysis_dir, set_name)
            self.image_layout_inner.addWidget(result_display)

        # 统计数据可视化
        self.plot_statistics()

    def plot_statistics(self):
        report_path = self.analyzer.report_path
        if not os.path.exists(report_path):
            QMessageBox.warning(self, "警告", "未生成分析报告。")
            return

        df = pd.read_csv(report_path)
        # 清理数据，移除'Overall'行
        df = df[df['Set'] != 'Overall']

        if df.empty:
            QMessageBox.warning(self, "警告", "分析报告中没有有效的数据。")
            return

        # 设置Matplotlib使用SimHei字体
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 创建图表
        fig = self.stats_canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(111)

        metrics = ['PSNR', 'SSIM']
        for metric in metrics:
            if metric in df.columns:
                ax1.plot(df['Set'], df[metric], marker='o', label=metric)

        ax1.set_xlabel('数据集')
        ax1.set_ylabel('值')
        ax1.set_title('PSNR 和 SSIM 统计')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # 检查是否有频域熵数据
        if 'Frequency Entropy' in df.columns:
            ax2 = ax1.twinx()
            ax2.bar(df['Set'], df['Frequency Entropy'], alpha=0.3, label='Frequency Entropy', color='gray')
            ax2.set_ylabel('频域熵')
            ax2.legend(loc='upper right')

        self.stats_canvas.draw()

    def append_log(self, msg):
        self.log_text.append(msg)

    # 添加拖放事件处理方法
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            dir_path = urls[0].toLocalFile()
            if os.path.isdir(dir_path):
                self.input_path.setText(dir_path)
                # 自动设置输出目录为 input_dir/output
                default_output = os.path.join(dir_path, "output")
                self.output_path.setText(default_output)
                # 自动加载图像
                self.load_images(dir_path)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
