import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import random

def generate_irregular_mask(size):
    mask = np.zeros(size, dtype=np.uint8)
    num_strokes = random.randint(5, 15)
    for _ in range(num_strokes):
        start_x = random.randint(0, size[1])
        start_y = random.randint(0, size[0])
        for _ in range(random.randint(1, 10)):
            angle = random.uniform(0, 2 * np.pi)
            length = random.randint(10, 60)
            brush_width = random.randint(10, 30)
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, brush_width)
            start_x, start_y = end_x, end_y
    return mask

def generate_box_mask(size):
    mask = np.zeros(size, dtype=np.uint8)
    num_boxes = random.randint(1, 5)
    for _ in range(num_boxes):
        x1 = random.randint(0, size[1] - 30)
        y1 = random.randint(0, size[0] - 30)
        w = random.randint(30, size[1] // 3)
        h = random.randint(30, size[0] // 3)
        x2 = min(x1 + w, size[1])
        y2 = min(y1 + h, size[0])
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask

def main(input_dir, output_dir, ext=".jpg", mask_type="mixed", preserve_names=False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob(f"*{ext}"))
    for img_path in image_paths:
        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        size = (h, w)

        # Randomly choose mask type
        if mask_type == "mixed":
            mask_kind = random.choice(["irregular", "box"])
        else:
            mask_kind = mask_type

        if mask_kind == "irregular":
            mask = generate_irregular_mask(size)
        elif mask_kind == "box":
            mask = generate_box_mask(size)
        else:
            raise ValueError("Unknown mask type")
        
        # Ensure binary mask: values are either 0 or 255
        mask = (mask > 127).astype(np.uint8) * 255

        # Save mask
        if preserve_names:
            mask_name = img_path.name  # exact same filename
        else:
            mask_name = img_path.stem + f"_mask{ext}"
        mask_path = output_dir / mask_name
        Image.fromarray(mask).save(mask_path)
        print(f"Saved mask to {mask_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to input mural images")
    parser.add_argument("output_dir", help="Where to save the masks")
    parser.add_argument("--ext", default=".jpg", help="Image file extension")
    parser.add_argument("--mask_type", default="mixed", choices=["irregular", "box", "mixed"])
    parser.add_argument("--preserve_names", action="store_true", help="Save masks with the same name as input images")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.ext, args.mask_type, args.preserve_names)
