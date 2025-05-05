import os
from PIL import Image

def resize_image(input_path, output_path, size, is_label=False):
    img = Image.open(input_path)
    if is_label:
        img = img.resize(size, Image.NEAREST)  # Keep mask binary
    else:
        img = img.resize(size, Image.BILINEAR)  # Smooth downscale for images
    img.save(output_path)

def resize_folder(input_dir, output_dir, size, is_label=False):
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.startswith('.'):
            continue
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        resize_image(input_path, output_path, size, is_label)
