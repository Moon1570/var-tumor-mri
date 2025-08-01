#extract_only_flair_slices_that_have_tumor.py
# This script extracts FLAIR slices from HDF5 files that contain tumor data,
# normalizes them, and saves both the grayscale images and overlays with tumor regions highlighted.

import h5py
import numpy as np
import imageio
import os
import cv2
from glob import glob

input_dir = "/Users/darklord/Research/VAR/code/var-mri/brats2020-kaggle/archive/BraTS2020_training_data/content/data"
output_img_dir = "tumor_slices/images"
output_overlay_dir = "tumor_slices/overlays"

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_overlay_dir, exist_ok=True)

h5_files = sorted(glob(os.path.join(input_dir, "*.h5")))

for h5_file in h5_files:
    with h5py.File(h5_file, 'r') as f:
        image = f['image'][:]   # shape (240, 240, 4)
        mask = f['mask'][:]     # shape (240, 240, 3)

        if np.any(mask > 0):  # skip slices without tumor
            flair = image[:, :, 3]
            flair = (flair - np.mean(flair)) / np.std(flair)
            flair = np.clip(flair, -5, 5)
            flair = ((flair - flair.min()) / (flair.max() - flair.min()) * 255).astype(np.uint8)

            # Save grayscale image
            filename = os.path.splitext(os.path.basename(h5_file))[0] + ".png"
            flair_path = os.path.join(output_img_dir, filename)
            imageio.imwrite(flair_path, flair)

            # Create overlay (red for tumor)
            mask_combined = np.sum(mask, axis=-1)
            overlay = cv2.cvtColor(flair, cv2.COLOR_GRAY2BGR)
            overlay[mask_combined > 0] = [255, 0, 0]  # Red overlay

            overlay_path = os.path.join(output_overlay_dir, filename)
            cv2.imwrite(overlay_path, overlay)

print("✅ Tumor slices and overlays saved.")
