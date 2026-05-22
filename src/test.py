"""
src/test.py

Inference / evaluation script for the Pix2Pix deblurring model.

Usage:
    # Basic (reads config.yaml)
    python src/test.py --config config.yaml --checkpoint outputs/checkpoints/ckpt_epoch_20.pth

    # Override output dir and number of examples to save
    python src/test.py --config config.yaml --checkpoint outputs/checkpoints/ckpt_epoch_20.pth --outdir outputs/final_results --max_save 200
"""

import os
import argparse
import yaml
from pathlib import Path
import csv
import time

import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

from dataset import get_dataloaders
from gan_model import Pix2PixModel

# optional metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
except Exception:
    compare_psnr = None
    compare_ssim = None


def read_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


def tensor_to_image(tensor):
    """Convert tensor in [-1,1] to uint8 HxWx3 numpy array."""
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    arr = (t.numpy() * 255.0).transpose(1, 2, 0).astype("uint8")
    return arr


def save_img(np_arr, path):
    Image.fromarray(np_arr).save(path)


def save_side_by_side(blur_t, fake_t, sharp_t, out_path, fname):
    blur_np = tensor_to_image(blur_t)
    fake_np = tensor_to_image(fake_t)
    sharp_np = tensor_to_image(sharp_t)
    b = Image.fromarray(blur_np)
    f = Image.fromarray(fake_np)
    s = Image.fromarray(sharp_np)
    w, h = b.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(b, (0, 0))
    canvas.paste(f, (w, 0))
    canvas.paste(s, (w * 2, 0))
    Path(out_path).mkdir(parents=True, exist_ok=True)
    canvas.save(os.path.join(out_path, fname))


def compute_metrics_np(pred_np, gt_np):
    psnr = None
    ssim = None
    if compare_psnr is not None:
        try:
            psnr = compare_psnr(gt_np, pred_np, data_range=255)
        except Exception:
            psnr = None
    if compare_ssim is not None:
        try:
            # multichannel=True for RGB
            ssim = compare_ssim(gt_np, pred_np, multichannel=True, data_range=255)
        except Exception:
            ssim = None
    return psnr, ssim


def test(config_path="config.yaml", checkpoint=None, outdir=None, max_save=500):
    cfg = read_config(config_path)

    data_root = cfg.get("data", {}).get("root", "data")
    dataset_name = cfg.get("data", {}).get("dataset_name", "LPBlur")
    img_size = cfg.get("data", {}).get("img_size", 256)

    training_cfg = cfg.get("training", {})
    batch_size = training_cfg.get("batch_size", 1)
    device = training_cfg.get("device", "cpu")
    num_workers = int(training_cfg.get("num_workers", 0))

    paths_cfg = cfg.get("paths", {})
    default_results = paths_cfg.get("results", "outputs/results")
    if outdir is None:
        outdir = default_results

    # create output dirs
    os.makedirs(outdir, exist_ok=True)
    metrics_csv = os.path.join(outdir, "test_metrics.csv")

    device_t = torch.device(device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu")
    print("Using device:", device_t)

    root_for_loader = os.path.join(data_root, dataset_name)
    _, _, test_loader = get_dataloaders(root=root_for_loader, img_size=img_size,
                                        batch_size=1, num_workers=num_workers, pin_memory=False)
    print("Test samples:", len(test_loader.dataset))

    # Build model
    model = Pix2PixModel(device=str(device_t),
                         lr_g=cfg.get("training", {}).get("lr_g", 2e-4),
                         lr_d=cfg.get("training", {}).get("lr_d", 2e-4),
                         lambda_L1=cfg.get("training", {}).get("lambda_L1", 100.0),
                         in_channels=3, out_channels=3,
                         ngf=cfg.get("model", {}).get("ngf", 32),
                         ndf=cfg.get("model", {}).get("ndf", 32))

    if checkpoint is None:
        raise ValueError("Please provide a checkpoint path with --checkpoint")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("Loading checkpoint:", checkpoint)
    model.load_checkpoint(checkpoint)

    # Run inference
    metrics = []
    saved = 0
    st = time.time()
    for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
        blur = batch["blur"].to(device_t)
        sharp = batch["sharp"].to(device_t)
        fname = batch.get("fname", [f"test_{i}.png"])[0]

        fake = model.generate(blur)

        # Save images
        # side-by-side (blur | fake | sharp)
        ss_name = f"{i:04d}_{fname}"
        save_side_by_side(blur[0], fake[0], sharp[0], outdir, ss_name)

        # also save just the deblurred output alone
        save_img(tensor_to_image(fake[0]), os.path.join(outdir, f"gen_{i:04d}_{fname}"))

        # compute metrics
        pred_np = tensor_to_image(fake[0])
        gt_np = tensor_to_image(sharp[0])
        psnr, ssim = compute_metrics_np(pred_np, gt_np)
        metrics.append({"fname": fname, "psnr": psnr if psnr is not None else "", "ssim": ssim if ssim is not None else ""})

        saved += 1
        if saved >= max_save:
            break

    elapsed = time.time() - st
    print(f"Finished. Saved {saved} examples. Time: {elapsed:.1f}s")

    # write metrics CSV
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fname", "psnr", "ssim"])
        for m in metrics:
            writer.writerow([m["fname"], m["psnr"], m["ssim"]])

    print("Metrics saved to:", metrics_csv)
    print("Individual results saved to:", outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--outdir", type=str, default=None, help="Where to save outputs")
    parser.add_argument("--max_save", type=int, default=500, help="Max number of test examples to save")
    args = parser.parse_args()

    test(config_path=args.config, checkpoint=args.checkpoint, outdir=args.outdir, max_save=args.max_save)
