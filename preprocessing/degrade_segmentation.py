import os
from resize_utils import resize_folder

# Settings
target_size = (256, 256)  # target downscaled resolution
root_in = "../MuralDH/Mural_seg"  # original path
root_out = "Mural_seg_downscaled" # output path, in current preprocessing dir

# Resize train split
resize_folder(os.path.join(root_in, "train", "images"),
              os.path.join(root_out, "train", "images"),
              target_size, is_label=False)

resize_folder(os.path.join(root_in, "train", "labels"),
              os.path.join(root_out, "train", "labels"),
              target_size, is_label=True)

# Resize test split
resize_folder(os.path.join(root_in, "test", "images"),
              os.path.join(root_out, "test", "images"),
              target_size, is_label=False)

resize_folder(os.path.join(root_in, "test", "labels"),
              os.path.join(root_out, "test", "labels"),
              target_size, is_label=True)

print("Done!")
