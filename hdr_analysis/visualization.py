# hdr_analysis/visualization.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import os

class Visualizer:
    def __init__(self, analysis_dir, font, logger=None):
        self.analysis_dir = analysis_dir
        self.font = font
        self.logger = logger or logging.getLogger('Visualizer')

    def generate_3d_brightness_plot(self, image, filename):
        """生成基于亮度的斜向3D图并保存"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rows, cols = gray.shape
            X, Y = np.meshgrid(range(cols), range(rows))
            Z = gray

            # 设置Matplotlib使用SimHei字体
            plt.rcParams['font.family'] = self.font.get_name()
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False)
            ax.set_title(f"{filename} 亮度3D图", fontproperties=self.font)
            ax.set_xlabel('X轴', fontproperties=self.font)
            ax.set_ylabel('Y轴', fontproperties=self.font)
            ax.set_zlabel('亮度', fontproperties=self.font)
            plot_path = os.path.join(self.analysis_dir, f"{filename}_亮度3D图.png")
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            self.logger.info(f"生成3D亮度图: {plot_path}")
            return plot_path
        except Exception as e:
            self.logger.error(f"生成3D亮度图失败: {filename}, 错误: {e}")
            return None

    def generate_difference_image(self, img1, img2, filename):
        """生成差异图像并保存"""
        try:
            difference = cv2.absdiff(img1, img2)
            diff_path = os.path.join(self.analysis_dir, f"{filename}_diff.png")
            cv2.imwrite(diff_path, difference)
            self.logger.info(f"生成差异图像: {diff_path}")
            return diff_path
        except Exception as e:
            self.logger.error(f"差异图像生成失败: {filename}, 错误: {e}")
            return None
