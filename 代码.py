import os
import re
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from functools import lru_cache
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numba
import sys

# 添加 Segment Anything 的路径
sys.path.append(r'D:\my data\segment anything\segment-anything-main')

# 高光谱文件解析的正则表达式
LINES_PATTERN = re.compile(r'lines', re.IGNORECASE)
SAMPLES_PATTERN = re.compile(r'samples', re.IGNORECASE)
DIGITS_PATTERN = re.compile(r'\d+')

# 自然排序函数
def natural_sort_key(s):
    """用于文件名的自然排序"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

@lru_cache(maxsize=128)
def parse_hdr(hdr_path):
    """解析 .hdr 文件，提取行数和样本数"""
    with open(hdr_path, 'r') as f:
        lines = f.read().splitlines()

    params = {'lines': None, 'samples': None}
    for line in lines:
        line_lower = line.lower()
        if LINES_PATTERN.search(line_lower):
            if match := DIGITS_PATTERN.search(line):
                params['lines'] = int(match.group())
        elif SAMPLES_PATTERN.search(line_lower):
            if match := DIGITS_PATTERN.search(line):
                params['samples'] = int(match.group())
    return params

# 模式配置
MODES = {
    'VisNIR': {'name': 'VisNIR', 'type_count': 1000, 'bands_to_extract': [452, 312, 172]},
    'NIR': {'name': 'NIR', 'type_count': 256, 'bands_to_extract': [76, 44, 11]},
    'Fluo': {'name': 'Fluo', 'type_count': 1000, 'bands_to_extract': [452, 312, 172]},
    'Raman': {'name': 'Raman', 'type_count': 1024, 'bands_to_extract': [289, 502, 715]}
}

# 选择模式（默认 VisNIR，可手动修改为 'NIR', 'Fluo', 或 'Raman'）
SELECTED_MODE = 'VisNIR'

# 验证模式
if SELECTED_MODE not in MODES:
    raise ValueError(f"无效的模式: {SELECTED_MODE}。可选模式: {list(MODES.keys())}")
mode = MODES[SELECTED_MODE]

class MaskProcessor:
    def __init__(self, image, mask_generator, input_filename=None):
        self.image = image
        self.mask_generator = mask_generator
        self.image_area = image.shape[0] * image.shape[1]
        self.area_threshold = self.image_area / 3
        self.input_filename = os.path.basename(input_filename) if input_filename else None

    def generate_and_filter_masks(self):
        """生成并初步过滤掩码"""
        masks = self.mask_generator.generate(self.image)
        filtered_masks = [mask for mask in masks if mask['area'] <= self.area_threshold]
        print(f"原始 mask 数量: {len(masks)}")
        print(f"初步过滤后 mask 数量: {len(filtered_masks)}")
        return filtered_masks

    def remove_overlapping_masks(self, masks):
        """逐一比较 IoU 去重掩码"""
        final_masks = []
        for i, mask in enumerate(masks):
            if all(self.compute_iou(mask['segmentation'], other['segmentation']) < 0.1 
                   for other in final_masks):
                final_masks.append(mask)
        print(f"去重后 mask 数量: {len(final_masks)}")
        return final_masks

    @staticmethod
    @numba.jit(nopython=True)
    def compute_iou(mask1, mask2):
        """使用 Numba 加速 IoU 计算"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    @staticmethod
    def calculate_brightness(mask, original_image):
        """计算掩码区域的亮度"""
        masked_image = original_image.copy()
        masked_image[~mask['segmentation']] = 0
        gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray[gray > 0]) / 255.0
        return brightness

    def filter_by_brightness(self, masks, brightness_threshold=0.7):
        """根据亮度过滤掩码"""
        if not masks:
            return []

        brightnesses = [(mask, self.calculate_brightness(mask, self.image)) for mask in masks]
        if not brightnesses:
            return []

        max_brightness_mask, max_brightness = max(brightnesses, key=lambda x: x[1])
        filtered = [mask for mask, brightness in brightnesses 
                    if abs(brightness - max_brightness) < (1 - brightness_threshold)]
        print(f"基于亮度过滤后 mask 数量: {len(filtered)}")
        return filtered

    def filter_by_average_area(self, masks, area_threshold_factor=0.5):
        """根据平均面积过滤掩码"""
        if not masks:
            return []

        areas = [mask['area'] for mask in masks]
        if not areas:
            return []

        avg_area = np.mean(areas)
        min_area = avg_area * (1 - area_threshold_factor)
        max_area = avg_area * (1 + area_threshold_factor)
        filtered = [mask for mask in masks if min_area <= mask['area'] <= max_area]
        print(f"基于平均面积过滤后 mask 数量: {len(filtered)}")
        return filtered

    def group_and_number_masks(self, masks, row_threshold=50, start_number=1):
        """分组并编号掩码，支持跨图像连续编号"""
        if not masks:
            return []

        boxes = []
        for mask in masks:
            y, x = np.where(mask['segmentation'])
            if len(x) == 0 or len(y) == 0:
                continue
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            boxes.append((x_min, y_min, x_max, y_max, mask))

        sorted_boxes = sorted(boxes, key=lambda b: b[1])
        if not sorted_boxes:
            return []

        rows = []
        current_row = [sorted_boxes[0]]
        prev_y = sorted_boxes[0][1]

        for box in sorted_boxes[1:]:
            if abs(box[1] - prev_y) > row_threshold:
                rows.append(sorted(current_row, key=lambda b: b[0]))
                current_row = []
            current_row.append(box)
            prev_y = box[1]
        rows.append(sorted(current_row, key=lambda b: b[0]))

        numbered_masks = []
        number = start_number
        for row in rows:
            for box in row:
                numbered_masks.append((box[4], number))
                number += 1

        return numbered_masks

    def create_visualization(self, numbered_masks, image):
        """创建掩码可视化图像"""
        img_overlay = np.ones((image.shape[0], image.shape[1], 4))
        img_overlay[:, :, 3] = 0
        visualized = image.copy()

        for mask, number in numbered_masks:
            m = mask['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img_overlay[m] = color_mask

            y, x = np.where(m)
            if len(x) > 0 and len(y) > 0:
                x_center = int(x.mean())
                y_center = int(y.mean())
                visualized = cv2.putText(
                    visualized,
                    str(number),
                    (x_center, y_center),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        visualized = (visualized * (1 - img_overlay[:, :, 3:4]) + 
                     img_overlay[:, :, :3] * 255 * img_overlay[:, :, 3:4]).astype(np.uint8)
        return visualized

    def show_anns(self, numbered_masks, image):
        """显示掩码可视化"""
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        img_overlay = np.ones((image.shape[0], image.shape[1], 4))
        img_overlay[:, :, 3] = 0

        for mask, number in numbered_masks:
            m = mask['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img_overlay[m] = color_mask

            y, x = np.where(m)
            if len(x) > 0 and len(y) > 0:
                x_center = int(x.mean())
                y_center = int(y.mean())
                plt.text(
                    x_center, y_center, str(number),
                    color='white', fontsize=12,
                    ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5)
                )

        ax.imshow(img_overlay)
        plt.axis('off')
        plt.show()

    def save_results(self, numbered_masks, save_path, filename=None):
        """保存掩码结果"""
        if filename is None:
            if self.input_filename:
                filename = os.path.splitext(os.path.basename(self.input_filename))[0]
            else:
                raise ValueError("必须提供输入文件名或保存文件名")

        os.makedirs(save_path, exist_ok=True)
        visualized_image = self.create_visualization(numbered_masks, self.image)
        output_file = os.path.join(save_path, f"{filename}_masks.jpg")
        cv2.imwrite(output_file, cv2.cvtColor(visualized_image, cv2.COLOR_RGB2BGR))
        print(f"结果已保存至: {output_file}")

    def apply_masks_to_hyperspectral(self, numbered_masks, hyper_data, output_dir, base_name):
        """将掩码应用于高光谱数据并保存种子数据，使用掩码编号"""
        try:
            lines, wide, type_count = hyper_data.shape
            seed_output_dir = os.path.join(output_dir, "seeds")
            os.makedirs(seed_output_dir, exist_ok=True)

            seed_count = 0
            for mask, number in numbered_masks:
                y, x = np.where(mask['segmentation'])
                if len(x) == 0 or len(y) == 0:
                    continue
                x_start, x_end = max(0, x.min()), min(wide, x.max() + 1)
                y_start, y_end = max(0, y.min()), min(lines, y.max() + 1)

                # 直接切片，避免复制整个数组
                seed_data = hyper_data[y_start:y_end, x_start:x_end, :]
                seed_mask = mask['segmentation'][y_start:y_end, x_start:x_end]
                # 在应用掩码前创建输出数组，减少内存占用
                output_data = np.zeros_like(seed_data)
                output_data[seed_mask] = seed_data[seed_mask]

                seed_filename = f"seed_{number}_{base_name}.npy"
                seed_path = os.path.join(seed_output_dir, seed_filename)
                np.save(seed_path, output_data, allow_pickle=False)
                seed_count += 1

                # 释放临时数组
                del seed_data, seed_mask, output_data

            print(f"已保存 {seed_count} 个种子的高光谱数据至: {seed_output_dir}")
            return seed_count

        except Exception as e:
            print(f"处理高光谱数据时出错: {str(e)}")
            return 0

def process_hyperspectral_to_rgb_and_masks(raw_file, hyperspectral_dir, rgb_dir, mask_dir, mask_generator, start_number, mode):
    """处理高光谱数据，生成RGB图像和掩码，根据模式设置 type_count 和 bands_to_extract"""
    file_name = raw_file[:-4]
    print(f'\n正在处理文件: {file_name} (模式: {mode["name"]})')

    filename_hdr = os.path.join(hyperspectral_dir, file_name + '.hdr')
    filename_raw = os.path.join(hyperspectral_dir, file_name + '.raw')

    if not os.path.exists(filename_hdr):
        print(f'.hdr文件不存在: {filename_hdr}，跳过该文件')
        return None, start_number - 1
    if not os.path.exists(filename_raw):
        print(f'.raw文件不存在: {filename_raw}，跳过该文件')
        return None, start_number - 1

    params = parse_hdr(filename_hdr)
    if not all([params['lines'], params['samples']]):
        print('无法从.hdr文件中提取lines或samples，跳过该文件')
        return None, start_number - 1

    lines, wide = params['lines'], params['samples']
    type_count = mode['type_count']
    bands_to_extract = mode['bands_to_extract']

    try:
        raw_data = np.fromfile(filename_raw, dtype='<u2')
        hyper_image = raw_data.reshape((lines, type_count, wide)).transpose(0, 2, 1)
        print(f'高光谱数据形状: {hyper_image.shape}')
    except Exception as e:
        print(f'读取RAW文件时出错: {e}，跳过该文件')
        return None, start_number - 1

    try:
        rgb_data = hyper_image[:, :, bands_to_extract]
    except IndexError as e:
        print(f'提取波段时出错: {e}，跳过该文件')
        return None, start_number - 1

    rgb_image = np.zeros((lines, wide, 3), dtype=np.uint8)
    for i in range(3):
        band = rgb_data[:, :, i].astype(np.float32)
        band_min = band.min()
        band_max = band.max()
        if band_max - band_min == 0:
            rgb_image[:, :, i] = 0
        else:
            normalized = (band - band_min) / (band_max - band_min + np.finfo(float).eps) * 255
            rgb_image[:, :, i] = normalized.astype(np.uint8)

    os.makedirs(rgb_dir, exist_ok=True)
    rgb_path = os.path.join(rgb_dir, f'RGB_from_hyperspectral_{file_name}_{mode["name"]}.png')
    Image.fromarray(rgb_image).save(rgb_path)
    print(f'RGB图像已保存到: {rgb_path}')

    image = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    processor = MaskProcessor(image, mask_generator, rgb_path)

    initial_masks = processor.generate_and_filter_masks()
    deduplicated_masks = processor.remove_overlapping_masks(initial_masks)
    brightness_filtered = processor.filter_by_brightness(deduplicated_masks)
    final_masks = processor.filter_by_average_area(brightness_filtered)
    numbered_masks = processor.group_and_number_masks(final_masks, start_number=start_number)

    processor.save_results(numbered_masks, mask_dir, filename=f'RGB_from_hyperspectral_{file_name}_{mode["name"]}')
    # processor.show_anns(numbered_masks, image)  # 取消注释以启用可视化

    seed_count = processor.apply_masks_to_hyperspectral(numbered_masks, hyper_image, mask_dir, f'{file_name}_{mode["name"]}')
    print(f"提取并保存了 {seed_count} 个种子的高光谱数据")

    # 计算最后编号
    last_number = start_number + len(numbered_masks) - 1 if numbered_masks else start_number - 1
    print(f"当前图像掩码编号: {[n for _, n in numbered_masks]}")
    return numbered_masks, last_number

# 目录配置
hyperspectral_dir = r"E:\outuchimei\outudusuchimeixitongvinir\矫正"
rgb_dir = r"D:\six article\code\MBKJJHW\RGB"
mask_dir = r"D:\six article\code\MBKJJHW\MASK"

# 初始化 SAM 掩码生成器
sam_checkpoint = r"D:\sam_vit_h_4b8939.pth"  # 确保路径正确
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "cuda"  # 如果无 GPU，改为 "cpu"
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# 获取并自然排序所有 .raw 文件
raw_files = [f for f in os.listdir(hyperspectral_dir) if f.endswith('.raw')]
if not raw_files:
    raise FileNotFoundError('指定路径下未找到.raw文件')
raw_files.sort(key=natural_sort_key)

# 逐个处理高光谱文件
current_number = 1
for raw_file in raw_files:
    numbered_masks, last_number = process_hyperspectral_to_rgb_and_masks(
        raw_file, hyperspectral_dir, rgb_dir, mask_dir, mask_generator, current_number, mode
    )
    if numbered_masks is not None:
        print(f"处理完成，mask 数量: {len(numbered_masks)}，最后编号: {last_number}")
        current_number = last_number + 1