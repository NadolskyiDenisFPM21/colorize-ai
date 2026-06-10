import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator.

    Input:  concatenated (L, ab) tensors — (B, 3, H, W)
    Output: (B, 1, P, P)  — patch-level real/fake predictions
    """

    def __init__(self):
        super().__init__()

        def _block(in_ch: int, out_ch: int, stride: int, use_batchnorm: bool = True) -> nn.Sequential:
            layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_batchnorm)]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            _block(3, 64, stride=2, use_batchnorm=False),   # 256 -> 128
            _block(64, 128, stride=2),                       # 128 -> 64
            _block(128, 256, stride=2),                      # 64  -> 32
            _block(256, 512, stride=1),                      # 32  -> 31
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # 31 -> 30
        )

    def forward(self, L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
        x = torch.cat([L, ab], dim=1)
        return self.model(x)
