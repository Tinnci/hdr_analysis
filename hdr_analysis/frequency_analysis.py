# hdr_analysis/frequency_analysis.py

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy.stats import entropy
import logging

class FrequencyAnalyzer:
    def __init__(self, analysis_dir, font, logger=None):
        self.analysis_dir = analysis_dir
        self.font = font
        self.logger = logger or logging.getLogger('FrequencyAnalyzer')

    def compute_power_spectrum(self, image):
        """计算图像的功率谱和幅度谱"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = fft2(gray)
            fshift = fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # +1避免log(0)
            power_spectrum = np.abs(fshift) ** 2
            return power_spectrum, magnitude_spectrum
        except Exception as e:
            self.logger.error(f"计算功率谱失败: {e}")
            return None, None

    def compute_edge_preservation(self, power_spectrum):
        """计算边缘保留指标，通过高频成分的比例来衡量"""
        try:
            rows, cols = power_spectrum.shape
            crow, ccol = rows // 2, cols // 2
            radius = min(crow, ccol) // 2
            mask = np.ones((rows, cols), np.uint8)
            cv2.circle(mask, (ccol, crow), radius, 0, -1)
            high_freq = power_spectrum * mask
            total_power = np.sum(power_spectrum)
            high_freq_power = np.sum(high_freq)
            edge_preservation = high_freq_power / total_power if total_power != 0 else 0
            return edge_preservation
        except Exception as e:
            self.logger.error(f"计算边缘保留指标失败: {e}")
            return None

    def compute_frequency_entropy(self, power_spectrum):
        """计算频域熵"""
        try:
            ps_norm = power_spectrum / np.sum(power_spectrum) if np.sum(power_spectrum) != 0 else power_spectrum
            ps_norm = ps_norm.flatten()
            freq_entropy = entropy(ps_norm + 1e-10)  # 加小值避免log(0)
            return freq_entropy
        except Exception as e:
            self.logger.error(f"计算频域熵失败: {e}")
            return None

    def save_spectrum_plot(self, magnitude_spectrum, filename):
        """保存功率谱图"""
        try:
            import matplotlib.pyplot as plt
            import os
            from matplotlib.font_manager import FontProperties

            plt.figure(figsize=(6,6))
            plt.imshow(magnitude_spectrum, cmap='gray')
            plt.title(f"{filename} 功率谱", fontproperties=self.font)
            plt.axis('off')
            ps_plot_path = os.path.join(self.analysis_dir, f"{filename}_功率谱.png")
            plt.savefig(ps_plot_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return ps_plot_path
        except Exception as e:
            self.logger.error(f"保存功率谱图失败: {e}")
            return None

    def save_entropy_plot(self, freq_entropy, filename):
        """保存频域熵图"""
        try:
            import matplotlib.pyplot as plt
            import os
            from matplotlib.font_manager import FontProperties

            plt.figure(figsize=(4,4))
            plt.text(0.5, 0.5, f'熵值: {freq_entropy:.2f}', fontsize=12, ha='center', fontproperties=self.font)
            plt.axis('off')
            entropy_plot_path = os.path.join(self.analysis_dir, f"{filename}_频域熵.png")
            plt.savefig(entropy_plot_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return entropy_plot_path
        except Exception as e:
            self.logger.error(f"保存频域熵图失败: {e}")
            return None
