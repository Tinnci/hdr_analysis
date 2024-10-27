# gui/main.py

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox,
    QCheckBox, QProgressBar, QMessageBox, QTabWidget,
    QScrollArea, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

from hdr_analysis.analyzer import HDRAnalyzer

class AnalyzerThread(QThread):
    progress = pyqtSignal(int)
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

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDR 图像分析工具")
        self.setGeometry(100, 100, 1200, 800)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.create_input_output_selection()
        self.create_options_selection()
        self.create_start_button()
        self.create_progress_bar()
        self.create_tabs()

        self.analyzer = None
        self.thread = None

    def create_input_output_selection(self):
        layout = QHBoxLayout()

        # 输入目录选择
        self.input_label = QLabel("输入目录:")
        self.input_path = QLabel("未选择")
        self.input_button = QPushButton("选择")
        self.input_button.clicked.connect(self.select_input_dir)

        layout.addWidget(self.input_label)
        layout.addWidget(self.input_path)
        layout.addWidget(self.input_button)

        # 输出目录选择
        self.output_label = QLabel("输出目录:")
        self.output_path = QLabel("未选择")
        self.output_button = QPushButton("选择")
        self.output_button.clicked.connect(self.select_output_dir)

        layout.addWidget(self.output_label)
        layout.addWidget(self.output_path)
        layout.addWidget(self.output_button)

        self.layout.addLayout(layout)

    def create_options_selection(self):
        layout = QHBoxLayout()

        # 图像类型选择
        self.image_type_label = QLabel("图像类型:")
        self.image_type_combo = QComboBox()
        self.image_type_combo.addItems(['original', 'fused', 'both'])

        layout.addWidget(self.image_type_label)
        layout.addWidget(self.image_type_combo)

        # 仅生成3D亮度图选择
        self.only_3d_checkbox = QCheckBox("仅生成斜向3D亮度图")

        layout.addWidget(self.only_3d_checkbox)

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
        self.layout.addWidget(self.progress_bar)

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
        self.image_layout_inner = QHBoxLayout()
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

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_path.setText(dir_path)

    def start_analysis(self):
        input_dir = self.input_path.text()
        output_dir = self.output_path.text()
        image_type = self.image_type_combo.currentText()
        only_3d = self.only_3d_checkbox.isChecked()

        if input_dir == "未选择" or output_dir == "未选择":
            QMessageBox.warning(self, "警告", "请先选择输入和输出目录。")
            return

        # 禁用按钮防止重复启动
        self.start_button.setEnabled(False)

        # 初始化HDRAnalyzer
        self.analyzer = HDRAnalyzer(
            input_dir=input_dir,
            output_dir=output_dir,
            report_path=os.path.join(output_dir, 'analysis_report.csv'),
            analysis_dir=os.path.join(output_dir, 'analysis_results'),
            image_type=image_type,
            only_3d=only_3d
        )

        # 启动分析线程
        self.thread = AnalyzerThread(self.analyzer)
        self.thread.finished.connect(self.analysis_finished)
        self.thread.error.connect(self.analysis_error)
        self.thread.start()

        # 显示进度条（此处简单模拟进度，可以根据实际情况调整）
        self.progress_bar.setRange(0, 0)  # 0表示无限循环进度条

    def analysis_finished(self, results):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "完成", "HDR图像分析已完成。")
        self.start_button.setEnabled(True)
        self.display_results(results)

    def analysis_error(self, error_msg):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        QMessageBox.critical(self, "错误", f"分析过程中出现错误:\n{error_msg}")
        self.start_button.setEnabled(True)

    def display_results(self, results):
        # 清空之前的图像
        for i in reversed(range(self.image_layout_inner.count())):
            item = self.image_layout_inner.itemAt(i)
            widget_to_remove = item.widget()
            if widget_to_remove:
                self.image_layout_inner.removeWidget(widget_to_remove)
                widget_to_remove.setParent(None)

        # 显示每个结果集的图像
        for result in results:
            set_name = result.get('Set', 'Unknown')
            # 显示差异图像和3D亮度图
            diff_path = os.path.join(self.analyzer.analysis_dir, f"{set_name}_diff.png")
            brightness_path = os.path.join(self.analyzer.analysis_dir, f"{set_name}_亮度3D图.png")
            power_spectrum_path = os.path.join(self.analyzer.analysis_dir, f"{set_name}_功率谱.png")
            entropy_plot_path = os.path.join(self.analyzer.analysis_dir, f"{set_name}_频域熵.png")

            # 创建垂直布局显示图片
            v_layout = QVBoxLayout()
            label_set = QLabel(f"数据集: {set_name}")
            label_set.setAlignment(Qt.AlignmentFlag.AlignCenter)
            v_layout.addWidget(label_set)

            # 差异图像
            if os.path.exists(diff_path):
                v_layout.addWidget(QLabel("差异图像:"))
                diff_pixmap = QPixmap(diff_path)
                label_diff = QLabel()
                label_diff.setPixmap(diff_pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
                label_diff.setAlignment(Qt.AlignmentFlag.AlignCenter)
                v_layout.addWidget(label_diff)
                # 添加缩放滑块
                slider_diff = QSlider(Qt.Orientation.Horizontal)
                slider_diff.setRange(50, 200)  # 缩放范围 50% - 200%
                slider_diff.setValue(100)
                slider_diff.valueChanged.connect(
                    lambda value, lbl=label_diff, pix=diff_pixmap: lbl.setPixmap(
                        pix.scaled(
                            pix.width() * value // 100,
                            pix.height() * value // 100,
                            Qt.AspectRatioMode.KeepAspectRatio
                        )
                    )
                )
                v_layout.addWidget(slider_diff)

            # 3D亮度图
            if os.path.exists(brightness_path):
                v_layout.addWidget(QLabel("3D亮度图:"))
                brightness_pixmap = QPixmap(brightness_path)
                label_brightness = QLabel()
                label_brightness.setPixmap(brightness_pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
                label_brightness.setAlignment(Qt.AlignmentFlag.AlignCenter)
                v_layout.addWidget(label_brightness)
                # 添加缩放滑块
                slider_brightness = QSlider(Qt.Orientation.Horizontal)
                slider_brightness.setRange(50, 200)
                slider_brightness.setValue(100)
                slider_brightness.valueChanged.connect(
                    lambda value, lbl=label_brightness, pix=brightness_pixmap: lbl.setPixmap(
                        pix.scaled(
                            pix.width() * value // 100,
                            pix.height() * value // 100,
                            Qt.AspectRatioMode.KeepAspectRatio
                        )
                    )
                )
                v_layout.addWidget(slider_brightness)

            # 功率谱图
            if os.path.exists(power_spectrum_path):
                v_layout.addWidget(QLabel("功率谱:"))
                ps_pixmap = QPixmap(power_spectrum_path)
                label_ps = QLabel()
                label_ps.setPixmap(ps_pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
                label_ps.setAlignment(Qt.AlignmentFlag.AlignCenter)
                v_layout.addWidget(label_ps)
                # 添加缩放滑块
                slider_ps = QSlider(Qt.Orientation.Horizontal)
                slider_ps.setRange(50, 200)
                slider_ps.setValue(100)
                slider_ps.valueChanged.connect(
                    lambda value, lbl=label_ps, pix=ps_pixmap: lbl.setPixmap(
                        pix.scaled(
                            pix.width() * value // 100,
                            pix.height() * value // 100,
                            Qt.AspectRatioMode.KeepAspectRatio
                        )
                    )
                )
                v_layout.addWidget(slider_ps)

            # 频域熵图
            if os.path.exists(entropy_plot_path):
                v_layout.addWidget(QLabel("频域熵:"))
                entropy_pixmap = QPixmap(entropy_plot_path)
                label_entropy = QLabel()
                label_entropy.setPixmap(entropy_pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio))
                label_entropy.setAlignment(Qt.AlignmentFlag.AlignCenter)
                v_layout.addWidget(label_entropy)
                # 添加缩放滑块
                slider_entropy = QSlider(Qt.Orientation.Horizontal)
                slider_entropy.setRange(50, 200)
                slider_entropy.setValue(100)
                slider_entropy.valueChanged.connect(
                    lambda value, lbl=label_entropy, pix=entropy_pixmap: lbl.setPixmap(
                        pix.scaled(
                            pix.width() * value // 100,
                            pix.height() * value // 100,
                            Qt.AspectRatioMode.KeepAspectRatio
                        )
                    )
                )
                v_layout.addWidget(slider_entropy)

            self.image_layout_inner.addLayout(v_layout)

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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
