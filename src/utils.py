"""
src/utils.py
Utility functions for image conversion, visualization, and metrics.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch

# Optional metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
except ImportError:
    compare_psnr = None
    compare_ssim = None


# -----------------------------------------
# Tensor <-> Image Conversion
# -----------------------------------------

def tensor_to_image(tensor):
    """
    Convert tensor [-1,1] -> uint8 numpy image (H,W,3)
    """
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    t = (t + 1) / 2
    t = t.clamp(0, 1)
    arr = (t.numpy() * 255).astype("uint8").transpose(1, 2, 0)
    return arr


# -----------------------------------------
# Save side-by-side comparison
# -----------------------------------------

def save_side_by_side(blur, fake, sharp, out_path, filename):
    """
    Saves a single image showing:
        blur | generated | sharp
    """
    blur_np = tensor_to_image(blur)
    fake_np = tensor_to_image(fake)
    sharp_np = tensor_to_image(sharp)

    b = Image.fromarray(blur_np)
    f = Image.fromarray(fake_np)
    s = Image.fromarray(sharp_np)

    w, h = b.size
    canvas = Image.new("RGB", (3 * w, h))
    canvas.paste(b, (0, 0))
    canvas.paste(f, (w, 0))
    canvas.paste(s, (2 * w, 0))

    Path(out_path).mkdir(parents=True, exist_ok=True)
    canvas.save(os.path.join(out_path, filename))


# -----------------------------------------
# Metrics: PSNR / SSIM
# -----------------------------------------

def compute_psnr(pred_np, gt_np):
    if compare_psnr is None:
        return None
    try:
        return compare_psnr(gt_np, pred_np, data_range=255)
    except:
        return None


def compute_ssim(pred_np, gt_np):
    if compare_ssim is None:
        return None
    try:
        return compare_ssim(gt_np, pred_np, multichannel=True, data_range=255)
    except:
        return None
