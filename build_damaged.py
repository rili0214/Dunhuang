import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

# 原始路径
image_dir = "/Users/yushen/Desktop/MuralDH/Mural_seg/test/images"
mask_dir = "/Users/yushen/Desktop/MuralDH/Mural_seg/test/labels"

# 输出路径
output_root = "/Users/yushen/Desktop/test_murals"
train_ratio = 0.8

# 创建输出文件夹
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_root, split, 'damaged'), exist_ok=True)
    os.makedirs(os.path.join(output_root, split, 'ground_truth'), exist_ok=True)

# 获取所有图像文件名
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
random.shuffle(image_filenames)

# 划分训练和验证
split_idx = int(len(image_filenames) * train_ratio)
train_filenames = image_filenames[:split_idx]
val_filenames = image_filenames[split_idx:]

def generate_damaged_image(image_path, mask_path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_np = np.array(image)
    mask_np = np.array(mask)
    mask_bin = (mask_np > 10).astype(np.uint8)[:, :, None]  # 转为 3通道广播用
    damaged_np = image_np * (1 - mask_bin)

    return image_np, damaged_np

# 执行处理
for split, filenames in [('train', train_filenames), ('val', val_filenames)]:
    for filename in tqdm(filenames, desc=f"Generating {split} set"):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if not os.path.exists(mask_path):
            continue

        gt_np, damaged_np = generate_damaged_image(img_path, mask_path)

        Image.fromarray(gt_np).save(os.path.join(output_root, split, 'ground_truth', filename))
        Image.fromarray(damaged_np.astype(np.uint8)).save(os.path.join(output_root, split, 'damaged', filename))

print("✅ Dataset generation complete.")
