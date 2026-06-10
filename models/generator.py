import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool = False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    """U-Net generator for image colorization.

    Input:  (B, 1, H, W)  — L channel in [-1, 1]
    Output: (B, 2, H, W)  — ab channels in [-1, 1]
    """

    def __init__(self):
        super().__init__()

        # Encoder: no batchnorm on first block
        self.enc1 = EncoderBlock(1, 64, use_batchnorm=False)    # 256 -> 128
        self.enc2 = EncoderBlock(64, 128)                        # 128 -> 64
        self.enc3 = EncoderBlock(128, 256)                       # 64  -> 32
        self.enc4 = EncoderBlock(256, 512)                       # 32  -> 16
        self.enc5 = EncoderBlock(512, 512)                       # 16  -> 8
        self.enc6 = EncoderBlock(512, 512)                       # 8   -> 4
        self.enc7 = EncoderBlock(512, 512)                       # 4   -> 2
        self.enc8 = EncoderBlock(512, 512, use_batchnorm=False)  # 2   -> 1  (bottleneck)

        # Decoder with skip connections; first 3 blocks use dropout
        self.dec1 = DecoderBlock(512, 512, use_dropout=True)     # 1   -> 2
        self.dec2 = DecoderBlock(1024, 512, use_dropout=True)    # 2   -> 4
        self.dec3 = DecoderBlock(1024, 512, use_dropout=True)    # 4   -> 8
        self.dec4 = DecoderBlock(1024, 512)                      # 8   -> 16
        self.dec5 = DecoderBlock(1024, 256)                      # 16  -> 32
        self.dec6 = DecoderBlock(512, 128)                       # 32  -> 64
        self.dec7 = DecoderBlock(256, 64)                        # 64  -> 128

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 2, kernel_size=4, stride=2, padding=1),  # 128 -> 256
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))

        return self.final(torch.cat([d7, e1], dim=1))
