"""
main.py
Unified runner for the GAN-Based License Plate Deblurring System.

Usage:
    # Train using config.yaml
    python main.py --mode train

    # Train overriding epochs
    python main.py --mode train --epochs 10

    # Resume training
    python main.py --mode train --resume outputs/checkpoints/ckpt_epoch_10.pth

    # Test a checkpoint
    python main.py --mode test --checkpoint outputs/checkpoints/ckpt_epoch_20.pth

    # Override test output directory
    python main.py --mode test --checkpoint outputs/checkpoints/ckpt_epoch_20.pth --outdir outputs/final_results
"""

# ------------------------------------------------------------
# 🔹 IMPORTANT FIX: Make sure "src/" is importable
# ------------------------------------------------------------
import os
import sys

# Add the src folder to Python path so imports like "from dataset import ..." work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import argparse

# Now these imports will work correctly
from src.train import train
from src.test import test


# ------------------------------------------------------------
# Main function
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GAN License Plate Deblurring System")

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "test"],
                        help="Mode to run: train or test")

    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")

    # Training-related optional args
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epoch count from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint")
    parser.add_argument("--save_every", type=int, default=None,
                        help="Override save_every value from config")

    # Test-related optional args
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load for testing")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Directory where results will be saved")
    parser.add_argument("--max_save", type=int, default=500,
                        help="Max number of test samples to save")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # TRAIN MODE
    # ------------------------------------------------------------
    if args.mode == "train":
        print("\n🚀 Starting TRAINING...\n")
        train(
            config_path=args.config,
            epochs=args.epochs,
            resume_ckpt=args.resume,
            save_every=args.save_every
        )

    # ------------------------------------------------------------
    # TEST MODE
    # ------------------------------------------------------------
    elif args.mode == "test":
        print("\n🔍 Starting TESTING...\n")

        if not args.checkpoint:
            print("❌ ERROR: --checkpoint is required in test mode.")
            sys.exit(1)

        test(
            config_path=args.config,
            checkpoint=args.checkpoint,
            outdir=args.outdir,
            max_save=args.max_save
        )

    print("\n✅ Done.\n")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
