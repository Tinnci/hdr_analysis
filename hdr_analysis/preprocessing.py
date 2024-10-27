# hdr_analysis/preprocessing.py

import cv2
import numpy as np
from PIL import Image, ExifTags
import logging

class Preprocessor:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger('Preprocessor')

    def correct_image_orientation(self, image_path):
        """纠正图像的方向并转换为OpenCV格式（BGR）"""
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
            return None

    def resize_image(self, image, target_size):
        """调整图像尺寸"""
        try:
            return cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        except Exception as e:
            self.logger.error(f"调整图像尺寸时发生错误: {e}")
            return image  # 返回原图像，避免后续处理失败
