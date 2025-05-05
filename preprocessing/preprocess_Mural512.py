import os
import random
import shutil
import subprocess
from PIL import Image
from resize_utils import resize_folder

import sys
sys.path.append(os.path.dirname(__file__))
from generate_dh_masks import main as generate_dh_masks

SEED = 42

def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Create output dirs if they don't exist, do nothing if exists
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'masks'), exist_ok=True)  # masks folder will be filled later

    all_images = [f for f in os.listdir(source_dir) if f.endswith('.png') and not f.startswith('.')]

    random.seed(SEED) # deterministic split
    random.shuffle(all_images) # shuffle images

    total = len(all_images)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    splits = {
        'train': all_images[:train_count],
        'val': all_images[train_count:train_count + val_count],
        'test': all_images[train_count + val_count:]
    }

    # Copy files to train/test/val folders
    for split, files in splits.items():
        for idx, file in enumerate(files):
            src = os.path.join(source_dir, file)
            new_name = f"{split}_{idx:04d}.png"  # like train_0001.png
            dst = os.path.join(output_dir, split, 'images', new_name)
            shutil.copy(src, dst)

    print(f"Dataset split Done! Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

def generate_masks_with_lama(images_dir, masks_dir, mask_config, lama_repo_dir):
    masks_temp_dir = masks_dir + "_temp"
    os.makedirs(masks_temp_dir, exist_ok=True)

    cmd = [
        'python', os.path.join(lama_repo_dir, 'bin/gen_mask_dataset.py'),
        mask_config,
        images_dir,
        masks_temp_dir,
        '--ext', 'png'
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = lama_repo_dir
    subprocess.run(cmd, check=True, env=env)

    # Move generated mask files
    for f in os.listdir(masks_temp_dir):
        if 'mask' in f and f.endswith('.png'):
            shutil.move(os.path.join(masks_temp_dir, f), os.path.join(masks_dir, f))

    shutil.rmtree(masks_temp_dir)
    print(f"Masks generated for {images_dir}")

def generate_masks_with_dh_script(images_dir, masks_dir):
    os.makedirs(masks_dir, exist_ok=True)
    generate_dh_masks(
        input_dir=images_dir,
        output_dir=masks_dir,
        ext=".png",              
        mask_type="mixed",        # or "irregular", "box"
        preserve_names=True
    )
    print(f"Masks generated for {images_dir} using Dunhuang-style masking.")


def preprocess_pipeline(source_dir, output_dir, mask_config, lama_repo_dir, resize_to=(256, 256), use_dh_mask=True):
    # Step 1: Split dataset
    split_dataset(source_dir, output_dir)

    # Step 2: For each split, generate masks and resize
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(output_dir, split, 'images')
        masks_dir = os.path.join(output_dir, split, 'masks')

        # Dunhuang mask or Lama mask
        if use_dh_mask:
            generate_masks_with_dh_script(images_dir, masks_dir)
        else:
            generate_masks_with_lama(images_dir, masks_dir, mask_config, lama_repo_dir)

        # Resize both images and masks
        resize_folder(images_dir, images_dir, size=resize_to, is_label=False)
        resize_folder(masks_dir, masks_dir, size=resize_to, is_label=True)

    print("Preprocessing pipeline done!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full preprocessing pipeline for Mural512 dataset.")
    parser.add_argument('--source_dir', type=str, required=True, help="Original Mural512 directory")
    parser.add_argument('--output_dir', type=str, required=True, help="Processed output directory")
    parser.add_argument('--mask_config', type=str, default='', help="Path to LaMa mask generator config (ignored if --use_dh_mask)")
    parser.add_argument('--lama_repo_dir', type=str, default='', help="Path to LaMa repo (ignored if --use_dh_mask)")
    parser.add_argument('--resize_to', type=int, nargs=2, required=True, help="Resize resolution as two integers, e.g., 256 256")
    parser.add_argument('--use_dh_mask', action='store_true', help="Use Dunhuang-style mask generator instead of LaMa")

    args = parser.parse_args()

    preprocess_pipeline(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        mask_config=args.mask_config,
        lama_repo_dir=args.lama_repo_dir,
        resize_to=tuple(args.resize_to),
        use_dh_mask=args.use_dh_mask
    )

