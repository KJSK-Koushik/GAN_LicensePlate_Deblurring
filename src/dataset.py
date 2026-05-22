"""
src/dataset.py

Usage:
    # Create train/val/test splits (copies files)
    python src/dataset.py --root data --dataset LPBlur --split --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1

    # Quick test that constructs DataLoaders using config.yaml
    python src/dataset.py --root data --dataset LPBlur --preview

This module:
 - Expects data/<dataset>/blur/ and data/<dataset>/sharp/ with matching filenames.
 - Can split into data/train/, data/val/, data/test/ directories (paired).
 - Provides PairDataset and get_dataloaders() for training/eval.
"""
import os
import argparse
import shutil
import random
from pathlib import Path
from typing import Tuple, List, Optional

import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm


# -------------------------
# Helper functions
# -------------------------
def read_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def list_images(folder: str, exts: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp")) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    return sorted([str(x.name) for x in p.iterdir() if x.suffix.lower() in exts])


def make_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -------------------------
# Splitting function
# -------------------------
def create_splits(root: str,
                  dataset_name: str,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> None:
    """
    Create train/val/test splits by copying files from dataset_name/blur and dataset_name/sharp
    into data/train, data/val and data/test (maintains pairing).
    """
    random.seed(seed)

    ds_root = Path(root) / dataset_name
    blur_dir = ds_root / "blur"
    sharp_dir = ds_root / "sharp"

    if not blur_dir.exists() or not sharp_dir.exists():
        raise FileNotFoundError(f"Couldn't find blur/sharp dirs under {ds_root}")

    file_list = list_images(blur_dir)
    if not file_list:
        raise FileNotFoundError(f"No images found in {blur_dir}")

    random.shuffle(file_list)
    n = len(file_list)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": file_list[:n_train],
        "val": file_list[n_train:n_train + n_val],
        "test": file_list[n_train + n_val:]
    }

    for split_name, files in splits.items():
        out_blur = Path(root) / split_name / "blur"
        out_sharp = Path(root) / split_name / "sharp"
        make_dir(out_blur)
        make_dir(out_sharp)
        print(f"Copying {len(files)} files to {split_name} ...")
        for fname in tqdm(files):
            src_b = blur_dir / fname
            src_s = sharp_dir / fname
            if not src_s.exists():
                # Try small filename differences (optional), but for now warn and skip
                print(f"Warning: matching sharp not found for {fname}, skipping.")
                continue
            shutil.copy2(src_b, out_blur / fname)
            shutil.copy2(src_s, out_sharp / fname)

    print("Splitting done. Train/Val/Test folders created under", root)


# -------------------------
# PyTorch Dataset
# -------------------------
class PairDataset(Dataset):
    """
    Expects folder structure:
        root/<split>/blur/*.jpg
        root/<split>/sharp/*.jpg

    Or you can pass split_folder directly (path to blur and sharp inside will be inferred).
    """
    def __init__(self,
                 root: str,
                 split: str = "train",
                 img_size: int = 256,
                 augment: bool = False):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.blur_dir = self.root / split / "blur"
        self.sharp_dir = self.root / split / "sharp"

        if not self.blur_dir.exists() or not self.sharp_dir.exists():
            raise FileNotFoundError(f"Expected {self.blur_dir} and {self.sharp_dir} to exist")

        self.filenames = list_images(self.blur_dir)
        # filter any without matching ground-truth
        self.filenames = [f for f in self.filenames if (self.sharp_dir / f).exists()]
        if len(self.filenames) == 0:
            raise RuntimeError("No paired images found for split: " + split)

        self.img_size = img_size
        self.augment = augment

        # transforms: PIL -> Tensor in [-1, 1]
        base_transforms = [
            transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1,1]
        ]

        if augment:
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
            ]
            self.transform = transforms.Compose(aug + base_transforms)
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        blur_path = self.blur_dir / fname
        sharp_path = self.sharp_dir / fname

        blur = Image.open(blur_path).convert("RGB")
        sharp = Image.open(sharp_path).convert("RGB")

        # apply same transform? torchvision transforms are deterministic per-call,
        # so call them separately; for random transforms it could differ.
        if self.augment:
            # For simple flips we can use the same RNG by seeding, but easiest is to randomly flip here:
            if random.random() > 0.5:
                blur = transforms.functional.hflip(blur)
                sharp = transforms.functional.hflip(sharp)

        blur_t = self.transform(blur)
        sharp_t = self.transform(sharp)

        return {"blur": blur_t, "sharp": sharp_t, "fname": fname}


# -------------------------
# DataLoader helper
# -------------------------
def get_dataloaders(root: str,
                    img_size: int = 256,
                    batch_size: int = 1,
                    num_workers: int = 0,
                    pin_memory: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train_loader, val_loader, test_loader
    """
    train_ds = PairDataset(root=root, split="train", img_size=img_size, augment=True)
    val_ds = PairDataset(root=root, split="val", img_size=img_size, augment=False)
    test_ds = PairDataset(root=root, split="test", img_size=img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="data", help="data root folder")
    parser.add_argument("--dataset", type=str, default="LPBlur", help="dataset folder name under root")
    parser.add_argument("--split", action="store_true", help="create train/val/test splits from dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--preview", action="store_true", help="build dataloaders and print sample shapes")
    parser.add_argument("--config", type=str, default="config.yaml", help="path to config file")
    args = parser.parse_args()

    if args.split:
        create_splits(root=args.root,
                      dataset_name=args.dataset,
                      train_ratio=args.train_ratio,
                      val_ratio=args.val_ratio,
                      test_ratio=args.test_ratio)

    if args.preview:
        cfg = read_config(args.config)
        img_size = cfg.get("data", {}).get("img_size", 256)
        batch_size = cfg.get("training", {}).get("batch_size", 1)
        num_workers = cfg.get("training", {}).get("num_workers", 0)
        root = args.root

        print("Building dataloaders with img_size=", img_size, "batch_size=", batch_size)
        t, v, te = get_dataloaders(root=root, img_size=img_size, batch_size=batch_size, num_workers=num_workers)
        print("Train samples:", len(t.dataset), "Val samples:", len(v.dataset), "Test samples:", len(te.dataset))
        # show one batch shapes
        batch = next(iter(t))
        print("Example batch keys:", batch.keys())
        print("Blur tensor shape:", batch["blur"].shape)
        print("Sharp tensor shape:", batch["sharp"].shape)
        print("Filenames sample:", batch["fname"][:5])

        # quick sanity check: save a few samples to outputs/preview
        preview_dir = Path("outputs/preview")
        make_dir(preview_dir)
        inv_norm = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])  # to convert [-1,1] to [0,1]
        for i in range(min(4, len(batch["blur"]))):
            b = batch["blur"][i]
            s = batch["sharp"][i]
            # convert to PIL and save
            b_img = transforms.ToPILImage()(inv_norm(b).clamp(0, 1))
            s_img = transforms.ToPILImage()(inv_norm(s).clamp(0, 1))
            fname = batch["fname"][i]
            b_img.save(preview_dir / f"preview_blur_{fname}")
            s_img.save(preview_dir / f"preview_sharp_{fname}")

        print("Saved previews to outputs/preview/")


if __name__ == "__main__":
    main()
