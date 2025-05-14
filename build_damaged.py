import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# 原始路径
root_dir = "/Users/yushen/Desktop/Mural512_processed"
split_dirs = ['train', 'val', 'test']

# 输出路径
output_root = "/Users/yushen/Desktop/Mural512"

# 创建输出文件夹
for split in split_dirs:
    os.makedirs(os.path.join(output_root, split, 'damaged'), exist_ok=True)
    os.makedirs(os.path.join(output_root, split, 'ground_truth'), exist_ok=True)

def get_image_mask_pairs(split_dir):
    """获取每个split目录下的image和mask文件对"""
    image_dir = os.path.join(root_dir, split_dir, 'images')
    mask_dir = os.path.join(root_dir, split_dir, 'masks')
    image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    image_mask_pairs = []
    for filename in image_filenames:
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        
        if os.path.exists(mask_path):
            image_mask_pairs.append((img_path, mask_path))
    
    return image_mask_pairs

# 获取各个split目录下的图片-标签对
for split in split_dirs:
    image_mask_pairs = get_image_mask_pairs(split)

    def generate_damaged_image(image_path, mask_path):
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_np = np.array(image)
        mask_np = np.array(mask)
        mask_bin = (mask_np > 10).astype(np.uint8)[:, :, None]  # 转为 3通道广播用
        damaged_np = image_np * (1 - mask_bin)

        return image_np, damaged_np

    # 执行处理
    for img_path, mask_path in tqdm(image_mask_pairs, desc=f"Generating {split} set"):
        gt_np, damaged_np = generate_damaged_image(img_path, mask_path)

        # 保存图像
        filename = os.path.basename(img_path)
        Image.fromarray(gt_np).save(os.path.join(output_root, split, 'ground_truth', filename))
        Image.fromarray(damaged_np.astype(np.uint8)).save(os.path.join(output_root, split, 'damaged', filename))

print("✅ Dataset generation complete.")
