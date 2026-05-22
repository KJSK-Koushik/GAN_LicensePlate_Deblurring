"""
src/generator.py

Corrected U-Net generator for paired image-to-image translation (blurry -> sharp).

This implementation fixes channel mismatches between decoder blocks and skip-concatenations.
Output range: [-1, 1] (Tanh).

Designed to be modest in size (ngf default = 32) for CPU training.
"""

import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=True, activation=True):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    if activation:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, kernel_size=4, stride=2, padding=1, norm=True, dropout=0.0):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.ReLU(inplace=True))
    if dropout and dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=32, use_dropout=False):
        """
        U-Net generator with careful channel sizing for decoder after skip-concat.
        ngf: base number of generator filters (32 recommended for CPU).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        self.use_dropout = use_dropout

        # Encoder (downsampling)
        # e1: ngf, e2: ngf*2, e3: ngf*4, e4: ngf*8, e5/e6/e7: ngf*8
        self.enc1 = conv_block(in_channels, ngf, norm=False)       # -> ngf
        self.enc2 = conv_block(ngf, ngf * 2)                       # -> ngf*2
        self.enc3 = conv_block(ngf * 2, ngf * 4)                   # -> ngf*4
        self.enc4 = conv_block(ngf * 4, ngf * 8)                   # -> ngf*8
        self.enc5 = conv_block(ngf * 8, ngf * 8)                   # -> ngf*8
        self.enc6 = conv_block(ngf * 8, ngf * 8)                   # -> ngf*8
        self.enc7 = conv_block(ngf * 8, ngf * 8)                   # -> ngf*8

        # bottleneck (reduce spatial to 1x1 if input appropriate)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (upsampling)
        # We must set in_ch for each decoder to match the concatenated tensor sizes used in forward:
        # pattern in forward:
        # d7 = dec7(bn) -> cat(d7,e7) -> passed into dec6
        # dec6 expects in_ch = dec7_out + e7_ch
        # and so on...
        # Using ngf=32 as base:
        # enc channels: e1=32,e2=64,e3=128,e4=256,e5=256,e6=256,e7=256
        # bottleneck output: 256

        # dec7: input is bottleneck (256) -> output 256
        self.dec7 = deconv_block(ngf * 8, ngf * 8, dropout=0.5 if use_dropout else 0.0)  # 256 -> 256

        # dec6: input is concat(dec7_out (256), e7 (256)) = 512 -> output 256
        self.dec6 = deconv_block(ngf * 8 * 2, ngf * 8, dropout=0.5 if use_dropout else 0.0)  # 512 -> 256

        # dec5: input concat(dec6_out (256), e6 (256)) = 512 -> output 256
        self.dec5 = deconv_block(ngf * 8 * 2, ngf * 8, dropout=0.5 if use_dropout else 0.0)  # 512 -> 256

        # dec4: input concat(dec5_out (256), e5 (256)) = 512 -> output 128 (ngf*4)
        self.dec4 = deconv_block(ngf * 8 * 2, ngf * 4)  # 512 -> 128

        # dec3: input concat(dec4_out (128), e4 (256)) = 384 -> output 64 (ngf*2)
        self.dec3 = deconv_block( (ngf * 4) + (ngf * 8), ngf * 2 )  # 384 -> 64

        # dec2: input concat(dec3_out (64), e3 (128)) = 192 -> output 32 (ngf)
        self.dec2 = deconv_block( (ngf * 2) + (ngf * 4), ngf )  # 192 -> 32

        # dec1: input concat(dec2_out (32), e2 (64)) = 96 -> output 32 (ngf)
        self.dec1 = deconv_block( (ngf) + (ngf * 2), ngf )  # 96 -> 32

        # final: concat(dec1_out (32), e1 (32)) = 64 -> upsample to out_channels
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Encoder forward
        e1 = self.enc1(x)   # [B, ngf, H/2, W/2]
        e2 = self.enc2(e1)  # [B, ngf*2, H/4, W/4]
        e3 = self.enc3(e2)  # [B, ngf*4, H/8, W/8]
        e4 = self.enc4(e3)  # [B, ngf*8, H/16, W/16]
        e5 = self.enc5(e4)  # [B, ngf*8, H/32, W/32]
        e6 = self.enc6(e5)  # [B, ngf*8, H/64, W/64]
        e7 = self.enc7(e6)  # [B, ngf*8, H/128, W/128]

        bn = self.bottleneck(e7)  # [B, ngf*8, H/256, W/256] (commonly 1x1 for 256x256 input)

        # Decoder with skip connections (apply deconv -> concat -> feed into next deconv)
        d7_up = self.dec7(bn)                 # -> [B, 256, ...]
        d7 = torch.cat([d7_up, e7], dim=1)    # -> [B, 512, ...]

        d6_up = self.dec6(d7)                 # -> [B, 256, ...]
        d6 = torch.cat([d6_up, e6], dim=1)    # -> [B, 512, ...]

        d5_up = self.dec5(d6)                 # -> [B, 256, ...]
        d5 = torch.cat([d5_up, e5], dim=1)    # -> [B, 512, ...]

        d4_up = self.dec4(d5)                 # -> [B, 128, ...]
        d4 = torch.cat([d4_up, e4], dim=1)    # -> [B, 384, ...]

        d3_up = self.dec3(d4)                 # -> [B, 64, ...]
        d3 = torch.cat([d3_up, e3], dim=1)    # -> [B, 192, ...]

        d2_up = self.dec2(d3)                 # -> [B, 32, ...]
        d2 = torch.cat([d2_up, e2], dim=1)    # -> [B, 96, ...]

        d1_up = self.dec1(d2)                 # -> [B, 32, ...]
        d1 = torch.cat([d1_up, e1], dim=1)    # -> [B, 64, ...]

        out = self.final(d1)                  # -> [B, out_channels, H, W]
        return out

    def _initialize_weights(self):
        """Initialize Conv layers with normal(0, 0.02) as in GAN papers."""
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
    # quick sanity check (only run manually)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    model = UNetGenerator(in_channels=3, out_channels=3, ngf=32).to(device)

    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Device:", device)
    print("Params:", count_params(model))

    x = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        y = model(x)
    print("Input:", x.shape, "Output:", y.shape, "min/max:", float(y.min()), float(y.max()))
