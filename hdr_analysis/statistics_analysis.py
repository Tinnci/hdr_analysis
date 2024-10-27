# hdr_analysis/statistics_analysis.py

import numpy as np
import pandas as pd
import logging
from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # 添加必要的导入

class StatisticsAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('StatisticsAnalyzer')

    def calculate_psnr(self, img1, img2):
        """计算PSNR"""
        try:
            return peak_signal_noise_ratio(img1, img2, data_range=img2.max() - img2.min())
        except Exception as e:
            self.logger.error(f"计算 PSNR 失败: {e}")
            return None

    def calculate_ssim(self, img1, img2):
        """计算SSIM"""
        try:
            return structural_similarity(img1, img2, multichannel=True)
        except Exception as e:
            self.logger.error(f"计算 SSIM 失败: {e}")
            return None

    def compute_statistics(self, image):
        """计算图像的统计指标"""
        try:
            stats = {}
            for i, color in enumerate(['Blue', 'Green', 'Red']):
                channel = image[:,:,i]
                stats[f'{color}_mean'] = np.mean(channel)
                stats[f'{color}_std'] = np.std(channel)
                stats[f'{color}_skew'] = pd.Series(channel.flatten()).skew()
                stats[f'{color}_kurt'] = pd.Series(channel.flatten()).kurt()
            return stats
        except Exception as e:
            self.logger.error(f"统计分析失败: {e}")
            return {}
