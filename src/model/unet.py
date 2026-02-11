"""
U-Net for wildfire spread probability prediction.

Input: 8-channel grid (terrain, fuel, weather, initial fire).
Output: 1-channel probability map (fire presence).
"""

import torch
import torch.nn as nn
from typing import Optional


class DoubleConv(nn.Module):
    """Two 3x3 convs, GroupNorm (AMD GPU compatible), ReLU, with optional dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout_rate: float = 0.0):
        super().__init__()
        # Use GroupNorm instead of BatchNorm (better AMD GPU compatibility)
        # GroupNorm with 8 groups works well for most channel sizes
        num_groups = min(8, out_ch)  # Ensure groups divides channels evenly
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    U-Net: 8 input channels -> 1 output channel (probability map).

    Encoder: 8 -> 64 -> 128 -> 256 -> 512
    Decoder: 512 -> 256 -> 128 -> 64 -> 1
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        base_channels: int = 64,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder with dropout
        self.enc1 = DoubleConv(in_channels, base_channels, dropout_rate)      # 64
        self.enc2 = DoubleConv(base_channels, base_channels * 2, dropout_rate)   # 128
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4, dropout_rate)  # 256
        self.enc4 = DoubleConv(base_channels * 4, base_channels * 8, dropout_rate)  # 512
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16, dropout_rate)  # 1024

        # Decoder with dropout
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = DoubleConv(base_channels * 16, base_channels * 8, dropout_rate)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4, dropout_rate)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2, dropout_rate)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels, dropout_rate)

        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


def build_unet(
    in_channels: int = 8,
    out_channels: int = 1,
    base_channels: Optional[int] = None,
    dropout_rate: float = 0.2,
) -> UNet:
    """Build U-Net from config-style args."""
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels or 64,
        dropout_rate=dropout_rate,
    )
