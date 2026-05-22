"""
src/gan_model.py

Pix2Pix GAN Model Wrapper:
- Loads Generator + Discriminator
- Defines GAN loss + L1 loss
- Provides training steps for both G and D
- Returns losses + predictions cleanly for train.py

This file makes training code clean and simple.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.generator import UNetGenerator
from src.discriminator import PatchDiscriminator

class Pix2PixModel:
    def __init__(self,
                 device="cpu",
                 lr_g=2e-4,
                 lr_d=2e-4,
                 lambda_L1=100.0,
                 in_channels=3,
                 out_channels=3,
                 ngf=32,
                 ndf=32):

        self.device = torch.device(device)

        # Initialize networks
        self.generator = UNetGenerator(
            in_channels=in_channels,
            out_channels=out_channels,
            ngf=ngf
        ).to(self.device)

        self.discriminator = PatchDiscriminator(
            in_channels=in_channels,
            ndf=ndf,
            n_layers=3
        ).to(self.device)

        # Losses
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        self.lambda_L1 = lambda_L1

        # Optimizers
        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # -------------------------------
    #  Discriminator Step
    # -------------------------------
    def train_discriminator_step(self, blur, sharp):
        """
        blur:  [B,3,H,W]
        sharp: [B,3,H,W] ground truth sharp image
        """

        self.discriminator.train()
        self.opt_d.zero_grad()

        # Real pairs (blur, sharp)
        pred_real = self.discriminator(blur, sharp)
        real_labels = torch.ones_like(pred_real, device=self.device)
        loss_real = self.adv_loss(pred_real, real_labels)

        # Fake pairs (blur, generator(blur))
        with torch.no_grad():
            fake_sharp = self.generator(blur)

        pred_fake = self.discriminator(blur, fake_sharp.detach())
        fake_labels = torch.zeros_like(pred_fake, device=self.device)
        loss_fake = self.adv_loss(pred_fake, fake_labels)

        # Total D loss
        d_loss = (loss_real + loss_fake) * 0.5
        d_loss.backward()
        self.opt_d.step()

        return d_loss.item()

    # -------------------------------
    #  Generator Step
    # -------------------------------
    def train_generator_step(self, blur, sharp):
        """
        G tries to:
         - fool D (adversarial loss)
         - match sharp image (L1 loss)
        """

        self.generator.train()
        self.opt_g.zero_grad()

        fake_sharp = self.generator(blur)

        # Adversarial loss (want D(blur, fake) -> real)
        pred_fake = self.discriminator(blur, fake_sharp)
        real_labels = torch.ones_like(pred_fake, device=self.device)
        loss_g_adv = self.adv_loss(pred_fake, real_labels)

        # L1 Reconstruction loss: |G(blur) - sharp|
        loss_g_l1 = self.l1_loss(fake_sharp, sharp) * self.lambda_L1

        # Total generator loss
        g_loss = loss_g_adv + loss_g_l1

        g_loss.backward()
        self.opt_g.step()

        return g_loss.item(), loss_g_adv.item(), loss_g_l1.item()

    # -------------------------------
    #  Inference (No Grad)
    # -------------------------------
    def generate(self, blur):
        """
        Forward pass only – used during validation/testing.
        Returns fake_sharp in [-1, 1]
        """
        self.generator.eval()
        with torch.no_grad():
            return self.generator(blur.to(self.device))

    # -------------------------------
    #  Save / Load
    # -------------------------------
    def save_checkpoint(self, path, epoch=None):
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "epoch": epoch  # optionally store epoch
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        self.opt_g.load_state_dict(ckpt["opt_g"])
        self.opt_d.load_state_dict(ckpt["opt_d"])
        epoch = ckpt.get("epoch", None)
        print(f"Loaded checkpoint from {path} (epoch={epoch})")
        return epoch

