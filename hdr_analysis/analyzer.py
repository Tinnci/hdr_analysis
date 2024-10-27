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

def analyze_set(subdir, input_dir, output_dir, image_type, only_3d, analysis_dir, font_path):
    """
    分析单个子文件夹中的图像集，并返回结果字典。
    这是一个顶层函数，便于被 ProcessPoolExecutor 序列化和调用。
    """
    # 设置日志
    logger = logging.getLogger('HDRAnalyzer')
    
    # 初始化各个模块
    preprocessor = Preprocessor(logger=logger)
    
    # 初始化字体
    if not os.path.exists(font_path):
        logger.error(f"指定的字体文件不存在: {font_path}")
        return {'Set': subdir, 'Error': f"字体文件不存在: {font_path}"}
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=font_path, size=12)
    
    freq_analyzer = FrequencyAnalyzer(analysis_dir=analysis_dir, font=font, logger=logger)
    stats_analyzer = StatisticsAnalyzer(logger=logger)
    visualizer = Visualizer(analysis_dir=analysis_dir, font=font, logger=logger)
    
    # 路径设置
    subdir_path = os.path.join(input_dir, subdir)
    fused_image_path_output = os.path.join(output_dir, f"{subdir}.jpg")

    result = {'Set': subdir}

    # 初始化处理标志
    process_fused = image_type in ['fused', 'both']
    process_original = image_type in ['original', 'both']

    # 处理融合图像
    if process_fused:
        # 检查融合图像是否存在
        if not os.path.isfile(fused_image_path_output):
            logger.warning(f"未找到对应的融合图像: {fused_image_path_output}")
        else:
            # 读取融合图像
            fused_image = preprocessor.correct_image_orientation(fused_image_path_output)
            if fused_image is None:
                logger.warning(f"无法读取融合图像: {fused_image_path_output}")
            else:
                if only_3d:
                    # 仅生成3D亮度图
                    visualizer.generate_3d_brightness_plot(fused_image, subdir)
                else:
                    # 检查输入图像的尺寸一致性
                    input_image_paths = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
                                        if f.lower().endswith(('.jpg','jpeg','png','tiff','bmp')) and '_fuse' not in f.lower()]
                    input_images = [preprocessor.correct_image_orientation(img_path) for img_path in input_image_paths]
                    input_images = [img for img in input_images if img is not None]
                    if not input_images:
                        logger.warning(f"子文件夹中未找到有效的输入图像: {subdir_path}")
                    else:
                        # 检查图像尺寸是否一致
                        input_sizes = [img.shape for img in input_images]
                        if len(set([size[:2] for size in input_sizes])) != 1:
                            logger.warning(f"子文件夹 {subdir} 中的输入图像尺寸不一致，进行统一调整。")
                            target_size = input_images[0].shape[:2]
                            input_images = [preprocessor.resize_image(img, target_size) for img in input_images]
                        else:
                            target_size = input_images[0].shape[:2]
                        
                        # 计算输入图像的平均图像
                        try:
                            avg_input = np.mean(np.array(input_images, dtype=np.float32), axis=0).astype(np.uint8)
                        except Exception as e:
                            logger.error(f"计算平均输入图像时发生错误: {e}")
                            avg_input = None
                        
                        if avg_input is not None:
                            # 检查融合图像尺寸是否一致
                            if fused_image.shape[:2] != avg_input.shape[:2]:
                                logger.warning(f"融合图像 {fused_image_path_output} 的尺寸 {fused_image.shape[:2]} 与平均输入图像的尺寸 {avg_input.shape[:2]} 不一致，进行调整。")
                                fused_image = preprocessor.resize_image(fused_image, avg_input.shape[:2])
                            
                            # 计算PSNR和SSIM
                            try:
                                psnr = stats_analyzer.calculate_psnr(avg_input, fused_image)
                                # 动态调整 win_size
                                min_side = min(fused_image.shape[:2])
                                win_size = min(7, min_side) if min_side >= 7 else 3  # 确保 win_size <= min_side
                                ssim = stats_analyzer.calculate_ssim(avg_input, fused_image, win_size=win_size)
                                result['PSNR'] = psnr
                                result['SSIM'] = ssim
                            except Exception as e:
                                logger.error(f"计算 PSNR/SSIM 失败: {e}")
                            
                            # 统计分析
                            input_stats = stats_analyzer.compute_statistics(avg_input)
                            output_stats = stats_analyzer.compute_statistics(fused_image)
                            
                            # 添加统计数据到结果
                            for key, value in input_stats.items():
                                result[f'Input_{key}'] = value
                            for key, value in output_stats.items():
                                result[f'Output_{key}'] = value
                            
                            # 频域分析
                            power_spectrum, magnitude_spectrum = freq_analyzer.compute_power_spectrum(fused_image)
                            if power_spectrum is not None:
                                edge_preservation = freq_analyzer.compute_edge_preservation(power_spectrum)
                                freq_entropy = freq_analyzer.compute_frequency_entropy(power_spectrum)
                                result['Edge Preservation'] = edge_preservation
                                result['Frequency Entropy'] = freq_entropy

                                # 保存频域分析图表
                                freq_analyzer.save_spectrum_plot(magnitude_spectrum, subdir)
                                freq_analyzer.save_entropy_plot(freq_entropy, subdir)

                            # 生成差异图像
                            if avg_input is not None:
                                visualizer.generate_difference_image(avg_input, fused_image, subdir)
                            
                            # 释放内存
                            del avg_input
                            del fused_image
                            del input_images
                            gc.collect()
    
    # 处理原始图像
    if process_original:
        # 获取原始图像（不包含 '_fuse' 的文件）
        original_image_paths = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
                                if f.lower().endswith(('.jpg','jpeg','png','tiff','bmp')) and '_fuse' not in f.lower()]
        original_images = [preprocessor.correct_image_orientation(img_path) for img_path in original_image_paths]
        original_images = [img for img in original_images if img is not None]
        if not original_images:
            logger.warning(f"子文件夹中未找到有效的原始输入图像: {subdir_path}")
        else:
            logger.info(f"在子文件夹 {subdir} 中找到 {len(original_images)} 张原始图像。")
            if only_3d:
                # 仅生成3D亮度图
                for idx, img in enumerate(original_images):
                    visualizer.generate_3d_brightness_plot(img, f"{subdir}_original_{idx+1}")
            else:
                # 如果需要在非仅3D模式下对原始图像执行分析，可以在这里添加
                pass  # 根据需求添加分析代码


            # 释放内存
            del original_images
            gc.collect()

    return result

class HDRAnalyzer:
    def __init__(self, input_dir, output_dir, report_path='analysis_report.csv', analysis_dir='analysis_results', max_workers=None, cpu_threshold=80.0, mem_threshold=80.0, image_type='both', only_3d=False, max_wait_time=300):
        """
        初始化HDRAnalyzer类。
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.report_path = report_path
        self.analysis_dir = analysis_dir
        self.image_type = image_type
        self.only_3d = only_3d
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

    def analyze(self):
        """执行HDR图像分析，包括频域分析和统计分析，或仅生成斜向3D亮度图。"""
        results = []
        sets_to_process = []

        # 获取所有子文件夹，排除output文件夹，并确保名称全为数字
        for subdir in os.listdir(self.input_dir):
            subdir_path = os.path.join(self.input_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue
            if subdir.lower() == 'output':
                continue
            if not subdir.isdigit():
                self.logger.warning(f"跳过非数字子文件夹: {subdir}")
                continue
            sets_to_process.append(subdir)

        self.logger.info(f"准备处理 {len(sets_to_process)} 个子文件夹。")

        # 设置中文字体路径
        font_path = os.path.join(os.path.dirname(__file__), '..', 'gui', 'resources', 'simhei.ttf')  # 修改字体路径
        if not os.path.exists(font_path):
            self.logger.error(f"指定的字体文件不存在: {font_path}")
            return

        try:
            # 使用 ProcessPoolExecutor 进行并行处理
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for subdir in sets_to_process:
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
                        self.logger.error(f"资源使用率仍高于阈值，跳过提交任务: {subdir}")
                        continue

                    # 提交任务
                    future = executor.submit(analyze_set, subdir, self.input_dir, self.output_dir, self.image_type, self.only_3d, self.analysis_dir, font_path)
                    futures[future] = subdir

                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sets"):
                    subdir = futures[future]
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"处理子文件夹 {subdir} 时发生未捕获的错误: {e}")
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

        if self.only_3d:
            # 仅生成3D亮度图时，不生成报告
            self.logger.info("仅生成3D亮度图，未生成分析报告。")
            self.results = []
            return

        # 计算总体统计
        try:
            overall = {
                'Set': 'Overall',
                'PSNR': np.nanmean([res['PSNR'] for res in results if res.get('PSNR') is not None]),
                'SSIM': np.nanmean([res['SSIM'] for res in results if res.get('SSIM') is not None]),
                'Edge Preservation': np.nanmean([res['Edge Preservation'] for res in results if res.get('Edge Preservation') is not None]),
                'Frequency Entropy': np.nanmean([res['Frequency Entropy'] for res in results if res.get('Frequency Entropy') is not None])
            }

            # 获取所有Input和Output统计键
            input_mean_keys = [key for key in results[0].keys() if key.startswith('Input_') and 'mean' in key]
            output_mean_keys = [key for key in results[0].keys() if key.startswith('Output_') and 'mean' in key]

            for key in input_mean_keys:
                overall[key] = np.nanmean([res[key] for res in results if key in res and res[key] is not None])
            for key in output_mean_keys:
                overall[key] = np.nanmean([res[key] for res in results if key in res and res[key] is not None])

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
