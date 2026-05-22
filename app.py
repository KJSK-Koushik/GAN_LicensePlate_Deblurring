import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as transforms

from src.gan_model import Pix2PixModel

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    model = Pix2PixModel(
        device="cpu",
        ngf=32,
        ndf=32
    )
    model.load_checkpoint("outputs/checkpoints/ckpt_epoch_30.pth")
    model.generator.eval()
    return model

model = load_model()

# ----------------------------
# Transform
# ----------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# ----------------------------
# Tensor → Image
# ----------------------------
def tensor_to_image(tensor):
    t = tensor.detach().cpu()
    t = (t + 1) / 2
    t = t.clamp(0, 1)
    t = t.numpy().transpose(1, 2, 0)
    return (t * 255).astype(np.uint8)

# ----------------------------
# UI
# ----------------------------
st.title("🚗 License Plate Deblurring System")
st.write("Upload a blurred license plate image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Optional sharpening
    image = image.filter(ImageFilter.SHARPEN)

    st.subheader("📥 Input Image")
    st.image(image, width=500)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model.generator(input_tensor)

    output_img = tensor_to_image(output[0])

    st.subheader("✨ Deblurred Output")
    st.image(output_img, width=500)

    st.subheader("🔍 Comparison")
    col1, col2 = st.columns(2)
    col1.image(image, caption="Blurred", width=300)
    col2.image(output_img, caption="Deblurred", width=300)