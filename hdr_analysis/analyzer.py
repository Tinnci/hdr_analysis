# hdr_analysis/analyzer.py

import os
import cv2
import numpy as np
import pandas as pd
import logging
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.fft import fft2, fftshift
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import gc
import psutil
import time
import sys

from .preprocessing import Preprocessor
from .frequency_analysis import FrequencyAnalyzer
from .statistics_analysis import StatisticsAnalyzer
from .visualization import Visualizer

def analyze_image_pair(original_image_path, fused_image_path, output_dir, analyses, font_path):
    """
    分析图像对，并返回结果字典。
    """
    # 设置日志
    logger = logging.getLogger('HDRAnalyzer')

    # 初始化各个模块
    preprocessor = Preprocessor(logger=logger)

    # 初始化字体
    if not os.path.exists(font_path):
        logger.error(f"指定的字体文件不存在: {font_path}")
        return {'Set': os.path.basename(original_image_path), 'Error': f"字体文件不存在: {font_path}"}
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=font_path, size=12)

    freq_analyzer = FrequencyAnalyzer(analysis_dir=output_dir, font=font, logger=logger)
    stats_analyzer = StatisticsAnalyzer(logger=logger)
    visualizer = Visualizer(analysis_dir=output_dir, font=font, logger=logger)

    filename = os.path.basename(original_image_path)
    set_name, _ = os.path.splitext(filename)
    result = {'Set': set_name}

    # 读取原始图像
    original_image = preprocessor.correct_image_orientation(original_image_path)
    if original_image is None:
        logger.warning(f"无法读取图像: {original_image_path}")
        return {'Set': set_name, 'Error': f"无法读取图像: {original_image_path}"}

    # 读取融合图像
    fused_image = preprocessor.correct_image_orientation(fused_image_path)
    if fused_image is None:
        logger.warning(f"无法读取图像: {fused_image_path}")
        return {'Set': set_name, 'Error': f"无法读取图像: {fused_image_path}"}

    if analyses.get('brightness_3d', False):
        # 生成3D亮度图
        visualizer.generate_3d_brightness_plot(fused_image, set_name)

    if analyses.get('statistical', False):
        # 统计分析
        original_stats = stats_analyzer.compute_statistics(original_image)
        fused_stats = stats_analyzer.compute_statistics(fused_image)
        for key, value in original_stats.items():
            result[f'Original_{key}'] = value
        for key, value in fused_stats.items():
            result[f'Fused_{key}'] = value

    if analyses.get('frequency', False):
        # 频域分析
        power_spectrum, magnitude_spectrum = freq_analyzer.compute_power_spectrum(fused_image)
        if power_spectrum is not None:
            edge_preservation = freq_analyzer.compute_edge_preservation(power_spectrum)
            freq_entropy = freq_analyzer.compute_frequency_entropy(power_spectrum)
            result['Edge Preservation'] = edge_preservation
            result['Frequency Entropy'] = freq_entropy

            # 保存频域分析图表
            freq_analyzer.save_spectrum_plot(magnitude_spectrum, set_name)
            freq_analyzer.save_entropy_plot(freq_entropy, set_name)

    if analyses.get('difference', False):
        # 生成差异图像（原始图像与融合图像的差异）
        visualizer.generate_difference_image(original_image, fused_image, set_name)

    if analyses.get('statistical', False) or analyses.get('frequency', False):
        # 计算PSNR和SSIM
        try:
            psnr = stats_analyzer.calculate_psnr(original_image, fused_image)
            ssim = stats_analyzer.calculate_ssim(original_image, fused_image)
            result['PSNR'] = psnr
            result['SSIM'] = ssim
        except Exception as e:
            logger.error(f"计算 PSNR/SSIM 失败: {e}")

    # 释放内存
    del original_image, fused_image
    gc.collect()

    return result

class HDRAnalyzer:
    def __init__(self, selected_images, output_dir, report_path='analysis_report.csv', analysis_dir='analysis_results', max_workers=None, cpu_threshold=80.0, mem_threshold=80.0, image_type='both', analyses=None, max_wait_time=300):
        """
        初始化HDRAnalyzer类。
        """
        self.selected_images = selected_images  # 接受一个图像列表
        self.output_dir = output_dir
        self.report_path = report_path
        self.analysis_dir = analysis_dir
        self.image_type = image_type
        self.analyses = analyses if analyses else {
            'statistical': False,
            'frequency': False,
            'difference': False,
            'brightness_3d': False
        }
        self.max_workers = max_workers or max(1, os.cpu_count() - 1)
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold
        self.max_wait_time = max_wait_time  # 最大等待时间（秒）
        os.makedirs(self.analysis_dir, exist_ok=True)

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(self.analysis_dir, 'analyzer.log'), mode='w', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger('HDRAnalyzer')

        self.results = []  # 添加此属性以存储分析结果

    def is_resource_available(self):
        """检查当前系统资源是否在允许范围内。"""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        mem_usage = psutil.virtual_memory().percent
        self.logger.debug(f"当前CPU使用率: {cpu_usage}%, 内存使用率: {mem_usage}%")
        return cpu_usage < self.cpu_threshold and mem_usage < self.mem_threshold

    def pair_images(self):
        """
        根据命名规则配对原始图像和融合图像。
        假设融合图像的命名为原始图像名后加'_fused'。
        例如，'image1.jpg'和'image1_fused.jpg'。
        """
        originals = {}
        fused = {}
        for img in self.selected_images:
            filename = os.path.basename(img)
            name, ext = os.path.splitext(filename)
            if name.endswith('_fused'):
                base_name = name[:-6]  # 去除'_fused'
                fused[base_name] = img
            else:
                originals[name] = img

        # 创建配对列表
        paired = []
        for name, orig_path in originals.items():
            fused_path = fused.get(name)
            if fused_path:
                paired.append((orig_path, fused_path))
            else:
                self.logger.warning(f"未找到融合图像对应的原始图像: {orig_path}")
        return paired

    def analyze(self):
        """执行HDR图像分析，包括频域分析和统计分析，或仅生成斜向3D亮度图。"""
        results = []
        if self.image_type == 'both':
            # 需要配对原始图像和融合图像
            paired_images = self.pair_images()
            self.logger.info(f"准备处理 {len(paired_images)} 对图像。")
        elif self.image_type == 'original':
            # 仅处理原始图像
            paired_images = [(img, img) for img in self.selected_images]
            self.logger.info(f"准备处理 {len(paired_images)} 张原始图像。")
        elif self.image_type == 'fused':
            # 仅处理融合图像
            paired_images = [(img, img) for img in self.selected_images]
            self.logger.info(f"准备处理 {len(paired_images)} 张融合图像。")
        else:
            self.logger.error(f"未知的image_type: {self.image_type}")
            return

        # 设置中文字体路径
        font_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'resources', 'simhei.ttf')  # 修改字体路径
        if not os.path.exists(font_path):
            self.logger.error(f"指定的字体文件不存在: {font_path}")
            return

        try:
            # 使用 ProcessPoolExecutor 进行并行处理
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for orig_path, fused_path in paired_images:
                    # 等待资源可用
                    wait_start_time = time.time()
                    while not self.is_resource_available():
                        elapsed_time = time.time() - wait_start_time
                        if elapsed_time > self.max_wait_time:
                            self.logger.error(f"等待资源可用超过最大等待时间 ({self.max_wait_time} 秒)，停止提交新任务。")
                            break
                        self.logger.debug("资源使用过高，等待...")
                        time.sleep(1)  # 等待1秒后重试

                    # 再次检查资源是否可用
                    if not self.is_resource_available():
                        self.logger.error(f"资源使用率仍高于阈值，跳过提交任务: {orig_path} 和 {fused_path}")
                        continue

                    # 提交任务
                    future = executor.submit(analyze_image_pair, orig_path, fused_path, self.analysis_dir, self.analyses, font_path)
                    futures[future] = (orig_path, fused_path)

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing image pairs"):
                    orig_path, fused_path = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"处理图像对 {orig_path} 和 {fused_path} 时发生未捕获的错误: {e}")
        except KeyboardInterrupt:
            self.logger.info("检测到键盘中断 (Ctrl+C)，正在尝试终止所有进程...")
            executor.shutdown(wait=False, cancel_futures=True)
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"在并行处理过程中发生错误: {e}")

        if not results:
            self.logger.warning("没有生成任何分析结果。")
            self.results = []
            return

        if not any(self.analyses.values()):
            # 如果没有任何分析选项被选中
            self.logger.info("未选择任何分析选项，未生成分析报告。")
            self.results = []
            return

        # 计算总体统计
        try:
            overall = {
                'Set': 'Overall'
            }

            # 统计分析
            if self.analyses.get('statistical', False):
                overall['PSNR'] = np.nanmean([res['PSNR'] for res in results if res.get('PSNR') is not None])
                overall['SSIM'] = np.nanmean([res['SSIM'] for res in results if res.get('SSIM') is not None])

                # 获取所有Original和Fused统计键
                if results:
                    stat_keys = [key for key in results[0].keys() if key.startswith('Original_') or key.startswith('Fused_')]
                    for key in stat_keys:
                        overall[key] = np.nanmean([res[key] for res in results if key in res and res[key] is not None])

            # 频域分析
            if self.analyses.get('frequency', False):
                overall['Edge Preservation'] = np.nanmean([res['Edge Preservation'] for res in results if res.get('Edge Preservation') is not None])
                overall['Frequency Entropy'] = np.nanmean([res['Frequency Entropy'] for res in results if res.get('Frequency Entropy') is not None])

            results.append(overall)
        except Exception as e:
            self.logger.error(f"计算总体统计时发生错误: {e}")

        # 保存报告
        try:
            df = pd.DataFrame(results)
            df.to_csv(self.report_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"分析完成，报告保存在: {self.report_path}")
            self.results = results  # 存储结果
        except Exception as e:
            self.logger.error(f"保存分析报告失败: {e}")
            self.results = []
