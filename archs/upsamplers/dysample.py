"""
DySample upsampler from 'Learning to Upsample by Learning to Sample'
https://arxiv.org/abs/2308.15085
Auto-detects device and falls back to PixelShuffle on MPS
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..base import register_upsampler


@register_upsampler('dysample')
class DySample(nn.Module):
    """Adapted from 'Learning to Upsample by Learning to Sample'
    Automatically uses PixelShuffle fallback on MPS devices
    """

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def __init__(
        self,
        in_channels: int,
        out_ch: int,
        scale: int = 2,
        groups: int = 4,
        end_convolution: bool = True,
    ) -> None:
        super().__init__()

        try:
            assert in_channels >= groups
            assert in_channels % groups == 0
        except:
            msg = "Incorrect in_channels and groups values."
            raise ValueError(msg)

        out_channels = 2 * groups * scale**2
        self.scale = scale
        self.groups = groups
        self.end_convolution = end_convolution
        self.in_channels = in_channels
        self.out_ch = out_ch

        # Always create PixelShuffle as fallback
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # DySample-specific layers (will be used on CUDA only)
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        if self.training:
            nn.init.trunc_normal_(self.offset.weight, std=0.02)
            nn.init.constant_(self.scope.weight, val=0)

        self.register_buffer("init_pos", self._init_pos())

        if end_convolution:
            # For DySample path, end_conv takes in_channels -> out_ch
            self.end_conv = nn.Conv2d(in_channels, out_ch, kernel_size=1)
        else:
            self.end_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check if we're on MPS
        if x.device.type == 'mps':
            # MPS fallback: use PixelShuffle directly
            # No end_conv needed as PixelShuffle already outputs correct channels
            output = self.pixel_shuffle(x)
            return output

        # CUDA path: use full DySample
        offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5

        mesh = torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
        coords = mesh.transpose(1, 2).unsqueeze(1).unsqueeze(0)
        coords = coords.type(x.dtype)

        normalizer = torch.tensor([W, H], device=x.device, dtype=x.dtype).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords_reshaped = coords.reshape(B, -1, H, W)
        coords_shuffled = F.pixel_shuffle(coords_reshaped, self.scale)
        coords = coords_shuffled.view(B, 2, -1, self.scale * H, self.scale * W)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous()
        coords = coords.flatten(0, 1)

        x_reshaped = x.reshape(B * self.groups, -1, H, W)

        output = F.grid_sample(
            x_reshaped,
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="zeros",
        )

        output = output.view(B, -1, self.scale * H, self.scale * W)

        if self.end_convolution and self.end_conv is not None:
            output = self.end_conv(output)

        return output
