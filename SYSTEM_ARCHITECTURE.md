# System Architecture

## 1. Overview

This project implements a Pix2Pix-style Generative Adversarial Network for license plate image deblurring. The system learns a paired image-to-image mapping from blurred license plate images to their corresponding sharp images.

The project has four major runtime paths:

1. Dataset preparation and loading
2. GAN training
3. Checkpoint-based testing and evaluation
4. Streamlit-based single-image inference

## 2. High-Level Architecture

```text
                  +----------------------+
                  |      config.yaml     |
                  | data/training/model  |
                  | paths configuration  |
                  +----------+-----------+
                             |
                             v
+----------------+    +-------------+    +-------------------+
| LPBlur Dataset | -> | PairDataset | -> | PyTorch DataLoaders|
| blur / sharp   |    | transforms  |    | train / val / test |
+----------------+    +-------------+    +---------+---------+
                                                   |
                                                   v
                                      +------------+-------------+
                                      |      Pix2PixModel        |
                                      | Generator + Discriminator|
                                      +------+-------------+-----+
                                             |             |
                                             v             v
                                   +--------------+  +--------------+
                                   | U-Net        |  | PatchGAN     |
                                   | Generator    |  | Discriminator|
                                   +------+-------+  +------+-------+
                                          |                 |
                                          v                 v
                                  +---------------+  +---------------+
                                  | Deblurred     |  | Real/Fake     |
                                  | Image         |  | Patch Logits  |
                                  +-------+-------+  +-------+-------+
                                          |                  |
                                          +---------+--------+
                                                    v
                                      +-------------+-------------+
                                      | Losses and Optimization   |
                                      | BCE adversarial + L1 loss |
                                      +-------------+-------------+
                                                    |
                                                    v
                                  +-----------------+-----------------+
                                  | outputs/checkpoints, val_results |
                                  +-----------------------------------+
```

## 3. Repository Components

| File | Responsibility |
| --- | --- |
| `main.py` | Unified command-line entry point for training and testing. |
| `app.py` | Streamlit demo app for uploading a blurred image and producing a deblurred output. |
| `config.yaml` | Central configuration for dataset paths, model size, training hyperparameters, and output directories. |
| `src/dataset.py` | Dataset splitting, paired image loading, preprocessing, augmentation, and DataLoader construction. |
| `src/generator.py` | U-Net generator that maps blurred RGB images to sharp RGB images. |
| `src/discriminator.py` | PatchGAN discriminator that classifies local blur/sharp image patches as real or fake. |
| `src/gan_model.py` | Pix2Pix wrapper that owns both networks, losses, optimizers, checkpoint save/load, and train steps. |
| `src/train.py` | Training loop, validation, metric logging, checkpointing, and sample image generation. |
| `src/test.py` | Checkpoint loading, test-set inference, metrics, and output image export. |
| `src/utils.py` | Shared image conversion, visualization, and metric helpers. |

## 4. Data Architecture

### Expected Dataset Layout

The loader expects paired images with matching filenames:

```text
data/
└── LPBlur/
    ├── train/
    │   ├── blur/
    │   └── sharp/
    ├── val/
    │   ├── blur/
    │   └── sharp/
    └── test/
        ├── blur/
        └── sharp/
```

`src/dataset.py` also includes a split utility that can copy images from:

```text
data/LPBlur/blur/
data/LPBlur/sharp/
```

into `train`, `val`, and `test` folders.

### Preprocessing Pipeline

Each image pair passes through the following steps:

1. Load blurred image and sharp target image using PIL.
2. Convert both images to RGB.
3. Resize both images to `img_size` from `config.yaml`, default `256 x 256`.
4. Convert images to tensors.
5. Normalize from `[0, 1]` to `[-1, 1]`.
6. During training only, optionally apply paired horizontal flipping.

Each dataset item returns:

```python
{
    "blur": blur_tensor,
    "sharp": sharp_tensor,
    "fname": filename
}
```

## 5. Model Architecture

### Generator: U-Net

The generator is defined in `src/generator.py` as `UNetGenerator`.

Purpose:

- Input: blurred RGB image tensor `[B, 3, H, W]`
- Output: deblurred RGB image tensor `[B, 3, H, W]`
- Output range: `[-1, 1]` through `Tanh`

Main design:

- Encoder: stacked convolution blocks with downsampling.
- Bottleneck: deepest compressed representation.
- Decoder: transposed convolution blocks with upsampling.
- Skip connections: encoder features are concatenated into decoder stages to preserve spatial detail.
- Normalization: `InstanceNorm2d`, suitable for small batch sizes.
- Activation:
  - Encoder uses `LeakyReLU`.
  - Decoder uses `ReLU`.
  - Final output uses `Tanh`.

Default generator scale:

```yaml
model:
  ngf: 32
```

With `img_size: 256`, the encoder progressively reduces the image until the bottleneck and then reconstructs it back to full resolution.

### Discriminator: PatchGAN

The discriminator is defined in `src/discriminator.py` as `PatchDiscriminator`.

Purpose:

- Input: paired image tensors, concatenated as `[blur, target]`
- Effective input channels: `6` for RGB pairs
- Output: patch-level logits `[B, 1, H_patch, W_patch]`

The discriminator receives either:

1. A real pair: blurred image + ground-truth sharp image.
2. A fake pair: blurred image + generated deblurred image.

Instead of judging the whole image with one scalar, PatchGAN predicts real/fake logits over local patches. This encourages sharper local texture and structure.

Default discriminator scale:

```yaml
model:
  ndf: 32
```

The final layer does not use sigmoid because training uses `BCEWithLogitsLoss` for numerical stability.

## 6. Training Architecture

Training is coordinated by `src/train.py` and `src/gan_model.py`.

### Training Entry Point

```bash
python main.py --mode train
```

Optional overrides:

```bash
python main.py --mode train --epochs 10
python main.py --mode train --resume outputs/checkpoints/ckpt_epoch_10.pth
```

### Training Flow

```text
main.py
  -> train(...)
      -> read config.yaml
      -> build train/val/test DataLoaders
      -> initialize Pix2PixModel
      -> optionally load checkpoint
      -> for each epoch:
           1. train discriminator
           2. train generator
           3. validate periodically
           4. append training metrics CSV
           5. save checkpoints periodically
```

### Discriminator Step

The discriminator learns to distinguish real and generated pairs:

```text
Real pair:
  D(blur, sharp) -> should be real

Fake pair:
  D(blur, G(blur)) -> should be fake
```

Loss:

```text
D_loss = 0.5 * (BCE(real_logits, 1) + BCE(fake_logits, 0))
```

### Generator Step

The generator learns to produce images that:

1. Fool the discriminator.
2. Stay close to the paired sharp ground truth.

Loss:

```text
G_loss = adversarial_loss + lambda_L1 * L1(fake_sharp, sharp)
```

Default reconstruction weight:

```yaml
training:
  lambda_L1: 100.0
```

This makes the model prioritize faithful deblurring while still using adversarial training to improve visual realism.

## 7. Validation, Metrics, and Outputs

During validation, the generator produces deblurred images for validation samples.

The system can compute:

- L1 validation loss
- PSNR, if `scikit-image` is available
- SSIM, if `scikit-image` is available

Validation images are saved as side-by-side comparisons:

```text
blur | generated | sharp
```

Configured output paths:

```yaml
paths:
  checkpoints: "./outputs/checkpoints"
  results: "./outputs/results"
  val_results: "./outputs/val_results"
```

Generated files include:

```text
outputs/checkpoints/ckpt_epoch_<N>.pth
outputs/checkpoints/training_log.csv
outputs/val_results/epoch_<N>_<sample>.png
```

These folders are ignored by Git because they contain generated artifacts and model files.

## 8. Testing and Evaluation Architecture

Testing is handled by `src/test.py`.

Command:

```bash
python main.py --mode test --checkpoint outputs/checkpoints/ckpt_epoch_30.pth
```

Flow:

```text
main.py
  -> test(...)
      -> read config.yaml
      -> build test DataLoader
      -> initialize Pix2PixModel
      -> load checkpoint
      -> generate deblurred image for each test sample
      -> save side-by-side result
      -> save generated-only result
      -> write PSNR/SSIM metrics CSV
```

Test outputs:

```text
outputs/results/
├── 0000_<filename>
├── gen_0000_<filename>
└── test_metrics.csv
```

## 9. Streamlit Inference Architecture

The demo app is defined in `app.py`.

Command:

```bash
streamlit run app.py
```

Flow:

```text
User uploads image
  -> PIL opens image and converts to RGB
  -> optional sharpening filter
  -> resize/crop/normalize transform
  -> Pix2PixModel loads checkpoint
  -> generator produces deblurred tensor
  -> tensor converted back to uint8 image
  -> Streamlit displays input, output, and comparison
```

The app currently loads:

```text
outputs/checkpoints/ckpt_epoch_30.pth
```

Because checkpoints are ignored by Git, this file must exist locally before running the app.

## 10. Configuration Architecture

`config.yaml` controls the main runtime behavior:

```yaml
data:
  root: "./data"
  dataset_name: "LPBlur"
  img_size: 256

training:
  device: "cpu"
  batch_size: 1
  num_epochs: 30
  lr_g: 0.0002
  lr_d: 0.0002
  lambda_L1: 100.0
  num_workers: 0
  validate_every: 2
  save_every: 5

model:
  ngf: 32
  ndf: 32

paths:
  checkpoints: "./outputs/checkpoints"
  results: "./outputs/results"
  val_results: "./outputs/val_results"
```

The defaults are CPU-friendly, with small generator and discriminator base filter counts.

## 11. Checkpoint Architecture

Checkpoints are saved by `Pix2PixModel.save_checkpoint`.

Each checkpoint contains:

```python
{
    "generator": generator_state_dict,
    "discriminator": discriminator_state_dict,
    "opt_g": generator_optimizer_state_dict,
    "opt_d": discriminator_optimizer_state_dict,
    "epoch": epoch
}
```

This allows:

- Resuming training from a saved epoch.
- Running inference with a trained generator.
- Keeping optimizer state for continued training.

## 12. End-to-End Data Flow

```text
Training:

blur image + sharp image
  -> PairDataset
  -> normalized tensors
  -> Generator creates fake sharp image
  -> Discriminator compares real pair and fake pair
  -> BCE adversarial loss + L1 reconstruction loss
  -> optimizer updates
  -> checkpoints and validation samples saved

Testing:

blur image + sharp image
  -> PairDataset
  -> load trained checkpoint
  -> Generator creates fake sharp image
  -> PSNR/SSIM metrics computed against sharp image
  -> generated and comparison images saved

Streamlit:

uploaded blurred image
  -> transform
  -> load trained checkpoint
  -> Generator creates deblurred image
  -> display result in browser
```

## 13. Deployment and Artifact Notes

The repository intentionally excludes:

- `data/`
- `outputs/`
- virtual environments
- Python caches
- model checkpoints such as `.pth`, `.pt`, `.ckpt`

This keeps GitHub focused on source code and documentation. Dataset files and trained model artifacts should be stored separately or regenerated locally.

## 14. Current Design Tradeoffs

- The project is configured for CPU-friendly training by default.
- Batch size defaults to `1`, which works well with `InstanceNorm2d`.
- The Streamlit app depends on a local checkpoint path.
- Training and testing duplicate some utility functions that also exist in `src/utils.py`; this works, but future cleanup could centralize those helpers.
- The generator and discriminator are lightweight enough for student/project use, but higher-quality deblurring may require larger models, longer training, GPU acceleration, and more data.
