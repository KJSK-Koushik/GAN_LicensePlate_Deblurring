"""
src/discriminator.py

PatchGAN discriminator (70x70-ish receptive field style) for paired image-to-image translation.
Designed to be lightweight so it runs acceptably on CPU / integrated GPU (Iris Xe).

Usage model layout:
 - By default the discriminator expects concatenated input: [blur, sharp] -> channels = 6 (3 + 3)
 - You may pass already-concatenated tensors or two tensors (input_img, target_img).
 - Last conv DOES NOT apply sigmoid so you can use BCEWithLogitsLoss for numerical stability.

Key choices for CPU-friendly training:
 - Uses InstanceNorm2d (better for batch_size=1)
 - Small number of filters (ndf=32 by default)
 - No final activation (logits returned)
"""

import torch
import torch.nn as nn
from typing import Optional


def conv_norm_lrelu(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN discriminator.

    Args:
        in_channels (int): channels per image (e.g., 3 for RGB). The network expects concatenated input
                           during forward (so effective input channels = in_channels * 2).
        ndf (int): base number of discriminator filters (default 32 for CPU).
        n_layers (int): number of downsampling layers (3 or 4 typical). Controls receptive field.
    """
    def __init__(self, in_channels: int = 3, ndf: int = 32, n_layers: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.ndf = ndf
        self.n_layers = n_layers

        # First layer - no normalization
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(in_channels * 2, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        # intermediate layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        # one more layer with stride=1 (to increase receptive field but keep spatial dims)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        # final conv -> 1 channel output (patch logits)
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)  # output is logits
        ]

        self.model = nn.Sequential(*sequence)
        self._initialize_weights()

    def forward(self, input_img: torch.Tensor, target_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        You can pass:
         - two tensors: input_img (blur) and target_img (sharp) -> they will be concatenated internally
         - a single tensor already concatenated along channel dim (channels = in_channels * 2)

        Returns:
         - logits tensor shape: [B, 1, H_patch, W_patch] (no sigmoid)
        """
        if target_img is not None:
            x = torch.cat([input_img, target_img], dim=1)
        else:
            x = input_img
        return self.model(x)

    def _initialize_weights(self):
        """Initialize Conv weights like standard GAN practice (normal(0, 0.02))."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == "__main__":
    # quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # default discriminator: takes concatenated (3+3) => 6 channels
    net = PatchDiscriminator(in_channels=3, ndf=32, n_layers=3).to(device)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Params: {:,}".format(count_params(net)))

    # dummy tensors
    # input image: Bx3x256x256 (blur), target image: Bx3x256x256 (sharp)
    x = torch.randn(1, 3, 256, 256).to(device)
    y = torch.randn(1, 3, 256, 256).to(device)
    out = net(x, y)
    print("Output shape (logits):", out.shape)  # expect [B,1,H_patch,W_patch]
    print("Min/Max logits:", float(out.min()), float(out.max()))
