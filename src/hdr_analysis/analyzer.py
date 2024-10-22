# src/hdr_analysis/analyzer.py

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from scipy.fft import fft2, fftshift
from scipy.fftpack import dct
import pandas as pd
import logging
from PIL import Image, ExifTags
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from tqdm import tqdm
import multiprocessing

class HDRAnalyzer:
    def __init__(self, input_dir, output_dir, report_path='analysis_report.csv', analysis_dir='analysis_results', max_workers=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.report_path = report_path
        self.analysis_dir = analysis_dir
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)  # 默认使用CPU核心数减1
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

    def correct_image_orientation(self, image_path):
        """
        使用Pillow读取图像并根据EXIF信息纠正方向，然后转换为OpenCV格式。
        """
        try:
            image = Image.open(image_path)
            # 获取EXIF方向标签
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif = image._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation, None)
                if orientation_value == 3:
                    image = image.rotate(180, expand=True)
                elif orientation_value == 6:
                    image = image.rotate(270, expand=True)
                elif orientation_value == 8:
                    image = image.rotate(90, expand=True)
            # 转换为OpenCV格式（BGR）
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            self.logger.warning(f"无法纠正图像方向或读取图像: {image_path}, 错误: {e}")
            # 如果无法读取或纠正方向，返回None
            return None

    def calculate_psnr(self, img1, img2):
        try:
            return compare_psnr(img1, img2, data_range=img2.max() - img2.min())
        except Exception as e:
            self.logger.error(f"计算 PSNR 失败: {e}")
            return None

    def calculate_ssim(self, img1, img2):
        min_dim = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
        win_size = min(11, min_dim)  # 设置窗口大小为11或图像最小边长
        if win_size % 2 == 0:
            win_size -= 1  # 确保窗口大小为奇数
        if win_size < 3:
            self.logger.warning(f"图像尺寸过小，无法计算SSIM: 最小窗口大小 {win_size} 小于3")
            return None
        try:
            return compare_ssim(img1, img2, win_size=win_size, channel_axis=-1)
        except ValueError as e:
            self.logger.error(f"SSIM计算时发生错误: {e}")
            return None

    def plot_histogram(self, image, title, save_path):
        color = ('b','g','r')
        plt.figure()
        plt.title(title)
        for i, col in enumerate(color):
            hist = cv2.calcHist([image],[i],None,[256],[0,256])
            plt.plot(hist, color = col, label=col.upper())
            plt.xlim([0,256])
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_frequency_spectrum(self, image, title, save_path):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        f = fft2(gray)
        fshift = fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift) + 1)  # +1避免log(0)

        plt.figure(figsize=(6,6))
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

    def perform_dct(self, image, title, save_path):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dct_transformed = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        magnitude_spectrum = np.log(np.abs(dct_transformed) + 1)

        plt.figure(figsize=(6,6))
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

    def compute_statistics(self, image):
        stats = {}
        for i, color in enumerate(['Blue', 'Green', 'Red']):
            channel = image[:,:,i]
            stats[f'{color}_mean'] = np.mean(channel)
            stats[f'{color}_std'] = np.std(channel)
            stats[f'{color}_skew'] = pd.Series(channel.flatten()).skew()
            stats[f'{color}_kurt'] = pd.Series(channel.flatten()).kurt()
        return stats

    def generate_difference_image(self, img1, img2, save_path):
        difference = cv2.absdiff(img1, img2)
        cv2.imwrite(save_path, difference)

    def resize_image(self, image, target_size):
        return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    def process_set(self, subdir):
        """
        处理单个子文件夹中的图像集，并返回结果字典。
        """
        subdir_path = os.path.join(self.input_dir, subdir)
        fused_image_path_set = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg','jpeg','png','tiff','bmp')) and '_fused' in f]
        if not fused_image_path_set:
            self.logger.warning(f"子文件夹 {subdir} 中未找到融合图像，跳过。")
            return None
        fused_image_path_set = os.path.join(subdir_path, fused_image_path_set[0])

        fused_image_path_output = os.path.join(self.output_dir, f"{subdir}.jpg")

        result = {
            'Set': subdir,
            'PSNR': None,
            'SSIM': None,
            # 统计数据将动态添加
        }

        # 读取融合图像（从set文件夹和output文件夹）
        fused_images = []
        for path in [fused_image_path_set, fused_image_path_output]:
            if os.path.isfile(path):
                fused_image = self.correct_image_orientation(path)
                if fused_image is not None:
                    fused_images.append(fused_image)
                else:
                    self.logger.warning(f"无法读取融合图像: {path}")
            else:
                self.logger.warning(f"未找到融合图像: {path}")

        if not fused_images:
            self.logger.warning(f"子文件夹 {subdir} 中未找到任何有效的融合图像，跳过。")
            return None

        # 只使用一个融合图像（优先使用set文件夹中的）
        fused_image = fused_images[0]

        # 读取输入图像（排除融合图像）
        input_image_paths = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path)
                            if f.lower().endswith(('.jpg','jpeg','png','tiff','bmp')) and '_fused' not in f]
        input_images = [self.correct_image_orientation(img_path) for img_path in input_image_paths]
        input_images = [img for img in input_images if img is not None]
        if not input_images:
            self.logger.warning(f"子文件夹中未找到有效的输入图像: {subdir_path}")
            return result

        # 检查所有输入图像的尺寸是否一致
        input_sizes = [img.shape for img in input_images]
        if len(set([size[:2] for size in input_sizes])) != 1:
            self.logger.warning(f"子文件夹 {subdir} 中的输入图像尺寸不一致，进行统一调整。")
            # 选择第一个图像的尺寸作为统一尺寸
            target_size = input_images[0].shape[:2]
            try:
                input_images = [self.resize_image(img, target_size) for img in input_images]
            except Exception as e:
                self.logger.error(f"调整输入图像尺寸时发生错误: {e}")
                return result
        else:
            target_size = input_images[0].shape[:2]

        # 计算输入图像的平均图像
        try:
            avg_input = np.mean(np.array(input_images, dtype=np.float32), axis=0).astype(np.uint8)
        except Exception as e:
            self.logger.error(f"计算平均输入图像时发生错误: {e}")
            return result

        # 检查融合图像尺寸是否与平均输入图像一致
        if fused_image.shape[:2] != avg_input.shape[:2]:
            self.logger.warning(f"融合图像 {fused_image_path_set} 的尺寸 {fused_image.shape[:2]} 与平均输入图像的尺寸 {avg_input.shape[:2]} 不一致，进行调整。")
            try:
                fused_image = self.resize_image(fused_image, avg_input.shape[:2])
            except Exception as e:
                self.logger.error(f"调整融合图像尺寸时发生错误: {e}")
                return result

        # 计算PSNR和SSIM
        try:
            psnr = self.calculate_psnr(avg_input, fused_image)
            ssim = self.calculate_ssim(avg_input, fused_image)
            result['PSNR'] = psnr
            result['SSIM'] = ssim
        except Exception as e:
            self.logger.error(f"计算 PSNR/SSIM 失败: {e}")

        # 统计分析
        try:
            input_stats = self.compute_statistics(avg_input)
            output_stats = self.compute_statistics(fused_image)
        except Exception as e:
            self.logger.error(f"统计分析失败: {e}")
            input_stats = {}
            output_stats = {}

        # 添加统计数据到结果
        for key, value in input_stats.items():
            result[f'Input_{key}'] = value
        for key, value in output_stats.items():
            result[f'Output_{key}'] = value

        # 频域分析
        try:
            fft_save_path = os.path.join(self.analysis_dir, f"{subdir}_fft.png")
            self.plot_frequency_spectrum(fused_image, f"{subdir} FFT Spectrum", fft_save_path)

            dct_save_path = os.path.join(self.analysis_dir, f"{subdir}_dct.png")
            self.perform_dct(fused_image, f"{subdir} DCT Spectrum", dct_save_path)
        except Exception as e:
            self.logger.error(f"频域分析失败: {e}")

        # 直方图
        try:
            hist_save_path_input = os.path.join(self.analysis_dir, f"{subdir}_input_hist.png")
            self.plot_histogram(avg_input, f"{subdir} Input Histogram", hist_save_path_input)

            hist_save_path_output = os.path.join(self.analysis_dir, f"{subdir}_output_hist.png")
            self.plot_histogram(fused_image, f"{subdir} Output Histogram", hist_save_path_output)
        except Exception as e:
            self.logger.error(f"直方图生成失败: {e}")

        # 差异图像
        try:
            diff_save_path = os.path.join(self.analysis_dir, f"{subdir}_diff.png")
            self.generate_difference_image(avg_input, fused_image, diff_save_path)
        except Exception as e:
            self.logger.error(f"差异图像生成失败: {e}")

        self.logger.info(f"已分析: {subdir}")
        return result

    def analyze(self):
        results = []
        sets_to_process = []

        # 获取所有子文件夹，排除output文件夹
        for subdir in os.listdir(self.input_dir):
            subdir_path = os.path.join(self.input_dir, subdir)
            if not os.path.isdir(subdir_path) or subdir.lower() == 'output':
                continue
            sets_to_process.append(subdir)

        self.logger.info(f"准备处理 {len(sets_to_process)} 个子文件夹。")

        # 使用 ProcessPoolExecutor 进行并行处理
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(self.process_set, subdir): subdir for subdir in sets_to_process}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing sets"):
                subdir = futures[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.error(f"处理子文件夹 {subdir} 时发生未捕获的错误: {e}")

        # 计算总体统计
        overall_psnr = np.nanmean([res['PSNR'] for res in results if res['PSNR'] is not None])
        overall_ssim = np.nanmean([res['SSIM'] for res in results if res['SSIM'] is not None])

        # 添加总体统计到结果
        overall_result = {
            'Set': 'Overall',
            'PSNR': overall_psnr,
            'SSIM': overall_ssim,
        }

        # 计算总体输入和输出统计的平均值
        input_mean_keys = [key for key in results[0].keys() if key.startswith('Input_') and 'mean' in key]
        output_mean_keys = [key for key in results[0].keys() if key.startswith('Output_') and 'mean' in key]

        for key in input_mean_keys:
            overall_result[key] = np.nanmean([res[key] for res in results if key in res])
        for key in output_mean_keys:
            overall_result[key] = np.nanmean([res[key] for res in results if key in res])

        results.append(overall_result)

        # 保存报告
        try:
            df = pd.DataFrame(results)
            df.to_csv(self.report_path, index=False)
            self.logger.info(f"分析完成，报告保存在: {self.report_path}")
        except Exception as e:
            self.logger.error(f"保存分析报告失败: {e}")
