# hdr_analysis/statistics_analysis.py

import numpy as np
import pandas as pd
import logging
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

class StatisticsAnalyzer:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('StatisticsAnalyzer')

    def calculate_psnr(self, img1, img2):
        """计算PSNR"""
        try:
            # 根据图像数据类型设置data_range
            if img1.dtype == np.uint8:
                data_range = 255
            elif img1.dtype == np.float32 or img1.dtype == np.float64:
                data_range = 1.0
            else:
                data_range = img2.max() - img2.min()
            return peak_signal_noise_ratio(img1, img2, data_range=data_range)
        except Exception as e:
            self.logger.error(f"计算 PSNR 失败: {e}")
            return None

    def calculate_ssim(self, img1, img2):
        """计算SSIM"""
        try:
            # structural_similarity的multichannel参数根据图像维度自动设置
            multichannel = True if img1.ndim == 3 else False
            return structural_similarity(img1, img2, multichannel=multichannel, data_range=img2.max() - img2.min())
        except Exception as e:
            self.logger.error(f"计算 SSIM 失败: {e}")
            return None

    def compute_statistics(self, image):
        """计算图像的统计指标"""
        try:
            stats = {}
            for i, color in enumerate(['Blue', 'Green', 'Red']):
                channel = image[:, :, i]
                stats[f'{color}_mean'] = np.mean(channel)
                stats[f'{color}_median'] = np.median(channel)
                stats[f'{color}_std'] = np.std(channel)
                stats[f'{color}_var'] = np.var(channel)
                stats[f'{color}_min'] = np.min(channel)
                stats[f'{color}_max'] = np.max(channel)
                stats[f'{color}_skew'] = pd.Series(channel.flatten()).skew()
                stats[f'{color}_kurt'] = pd.Series(channel.flatten()).kurt()
            return stats
        except Exception as e:
            self.logger.error(f"统计分析失败: {e}")
            return {}
