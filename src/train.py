"""
src/train.py

Training script for Pix2Pix-style GAN (blurry -> sharp)
Supports clean pause & resume from checkpoints.
"""

import os
import argparse
import time
from pathlib import Path
import yaml
import csv

import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

from dataset import get_dataloaders
from gan_model import Pix2PixModel

# optional metrics
try:
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
    from skimage.metrics import structural_similarity as compare_ssim
except Exception:
    compare_psnr = None
    compare_ssim = None


# -------------------------------------------------
# Helper: Read config
# -------------------------------------------------
def read_config(path="config.yaml"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


# -------------------------------------------------
# Tensor -> Image
# -------------------------------------------------
def tensor_to_image(tensor):
    t = tensor.detach().cpu()
    if t.dim() == 4:
        t = t[0]
    t = (t + 1.0) / 2.0
    t = t.clamp(0, 1)
    return (t.numpy() * 255).astype("uint8").transpose(1, 2, 0)


# -------------------------------------------------
# Save comparison image
# -------------------------------------------------
def save_side_by_side(blur, fake, sharp, out_dir, name):
    b = Image.fromarray(tensor_to_image(blur))
    f = Image.fromarray(tensor_to_image(fake))
    s = Image.fromarray(tensor_to_image(sharp))

    w, h = b.size
    canvas = Image.new("RGB", (w * 3, h))
    canvas.paste(b, (0, 0))
    canvas.paste(f, (w, 0))
    canvas.paste(s, (w * 2, 0))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    canvas.save(os.path.join(out_dir, name))


# -------------------------------------------------
# Metrics
# -------------------------------------------------
def compute_metrics_np(pred, gt):
    psnr = None
    ssim = None

    if compare_psnr:
        try:
            psnr = compare_psnr(gt, pred, data_range=255)
        except Exception:
            psnr = None

    # SSIM is optional and fragile for small images → protect it
    if compare_ssim:
        try:
            h, w = gt.shape[:2]
            if h >= 7 and w >= 7:
                ssim = compare_ssim(
                    gt,
                    pred,
                    multichannel=True,
                    data_range=255
                )
        except Exception:
            ssim = None

    return psnr, ssim



# -------------------------------------------------
# Training
# -------------------------------------------------
def train(config_path="config.yaml", epochs=None, resume_ckpt=None, save_every=None):
    cfg = read_config(config_path)

    # ---------------- config ----------------
    data_root = cfg.get("data", {}).get("root", "data")
    dataset_name = cfg.get("data", {}).get("dataset_name", "LPBlur")
    img_size = cfg.get("data", {}).get("img_size", 256)

    tcfg = cfg.get("training", {})
    device = tcfg.get("device", "cpu")
    batch_size = tcfg.get("batch_size", 1)
    num_epochs = int(epochs) if epochs else int(tcfg.get("num_epochs", 30))
    lr_g = tcfg.get("lr_g", 2e-4)
    lr_d = tcfg.get("lr_d", 2e-4)
    lambda_L1 = tcfg.get("lambda_L1", 100.0)
    validate_every = int(tcfg.get("validate_every", 1))
    num_workers = int(tcfg.get("num_workers", 0))

    paths = cfg.get("paths", {})
    ckpt_dir = paths.get("checkpoints", "outputs/checkpoints")
    val_dir = paths.get("val_results", "outputs/val_results")

    if save_every is None:
        save_every = int(tcfg.get("save_every", 5))

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    device = torch.device(device)
    print(f"Using device: {device}")

    # ---------------- data ----------------
    root = os.path.join(data_root, dataset_name)
    train_loader, val_loader, _ = get_dataloaders(
        root=root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    print(f"Dataset sizes — Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    # ---------------- model ----------------
    model = Pix2PixModel(
        device=str(device),
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_L1=lambda_L1,
        ngf=cfg.get("model", {}).get("ngf", 32),
        ndf=cfg.get("model", {}).get("ndf", 32)
    )

    # ---------------- resume ----------------
    start_epoch = 1
    if resume_ckpt and os.path.exists(resume_ckpt):
        print(f"Resuming from checkpoint: {resume_ckpt}")
        loaded_epoch = model.load_checkpoint(resume_ckpt)

        if loaded_epoch is not None:
            start_epoch = loaded_epoch + 1
        else:
            # fallback: infer from filename
            try:
                ep = int(os.path.basename(resume_ckpt).split("_")[-1].split(".")[0])
                start_epoch = ep + 1
                print(f"Inferred epoch {ep} from filename")
            except Exception:
                start_epoch = 1

        print(f"Resuming from epoch {start_epoch}")

    # ---------------- logging ----------------
    log_csv = os.path.join(ckpt_dir, "training_log.csv")
    if not os.path.exists(log_csv):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "d_loss", "g_loss", "adv", "l1", "val_l1", "psnr", "ssim", "time"]
            )

    # ---------------- training loop ----------------
    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()
        model.generator.train()
        model.discriminator.train()

        d_sum = g_sum = adv_sum = l1_sum = 0.0
        batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)

            d = model.train_discriminator_step(blur, sharp)
            g, adv, l1 = model.train_generator_step(blur, sharp)

            d_sum += d
            g_sum += g
            adv_sum += adv
            l1_sum += l1
            batches += 1

        d_avg = d_sum / batches
        g_avg = g_sum / batches
        adv_avg = adv_sum / batches
        l1_avg = l1_sum / batches

        # ---------------- validation ----------------
        val_l1 = psnr = ssim = None
        if epoch % validate_every == 0:
            model.generator.eval()
            l1s, psnrs, ssims = [], [], []

            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    blur = batch["blur"].to(device)
                    sharp = batch["sharp"].to(device)
                    fake = model.generate(blur)

                    l1s.append(torch.nn.functional.l1_loss(fake, sharp).item())

                    p, s = compute_metrics_np(
                        tensor_to_image(fake),
                        tensor_to_image(sharp)
                    )
                    if p: psnrs.append(p)
                    if s: ssims.append(s)

                    if i < 5:
                        save_side_by_side(
                            blur[0], fake[0], sharp[0],
                            val_dir, f"epoch_{epoch:03d}_{i}.png"
                        )

            val_l1 = float(np.mean(l1s))
            psnr = float(np.mean(psnrs)) if psnrs else None
            ssim = float(np.mean(ssims)) if ssims else None

        elapsed = time.time() - start_time

        print(f"Epoch {epoch}/{num_epochs} | D {d_avg:.4f} | G {g_avg:.4f} | Adv {adv_avg:.4f} | L1 {l1_avg:.4f}")
        if val_l1:
            print(f"  Val L1 {val_l1:.4f} | PSNR {psnr} | SSIM {ssim}")

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, d_avg, g_avg, adv_avg, l1_avg, val_l1, psnr, ssim, elapsed]
            )

        if epoch % save_every == 0 or epoch == num_epochs:
            path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pth")
            model.save_checkpoint(path, epoch=epoch)
            print(f"Saved checkpoint: {path}")

    print("Training completed.")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    args = parser.parse_args()

    train(args.config, args.epochs, args.resume, args.save_every)
