# GAN License Plate Deblurring

GAN-based license plate deblurring project with training, testing, and a Streamlit demo app.

## Project Structure

- `main.py` - command-line runner for training and testing.
- `app.py` - Streamlit app for uploading and deblurring images.
- `config.yaml` - training, model, dataset, and output path configuration.
- `src/` - dataset loading, GAN model, generator, discriminator, training, testing, and utility code.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train the model:

```bash
python main.py --mode train
```

Resume training:

```bash
python main.py --mode train --resume outputs/checkpoints/ckpt_epoch_10.pth
```

Test a checkpoint:

```bash
python main.py --mode test --checkpoint outputs/checkpoints/ckpt_epoch_30.pth
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Data and Checkpoints

The `data/` and `outputs/` directories are intentionally ignored by Git because they can contain datasets, generated images, and model checkpoints. Place the LPBlur dataset under `data/LPBlur` and trained checkpoints under `outputs/checkpoints`.
