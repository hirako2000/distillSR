"""
RealPLKSR architecture from neosr framework
"""

from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .blocks.plk_block import PLKBlock
from .upsamplers.dysample import DySample


class realplksr(nn.Module):
    """Partial Large Kernel CNNs for Efficient Super-Resolution"""

    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        dim: int = 64,
        n_blocks: int = 28,
        upscaling_factor: int = 4,
        kernel_size: int = 17,
        split_ratio: float = 0.25,
        use_ea: bool = True,
        norm_groups: int = 4,
        dropout: float = 0,
        dysample: bool = False,
        res_scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        self.upscale = upscaling_factor
        self.dysample = dysample
        if not self.training:
            dropout = 0

        # Build feats sequentially
        layers = []

        # Initial convolution
        layers.append(nn.Conv2d(in_ch, dim, 3, 1, 1))

        # PLKBlocks with optional residual scaling
        for _ in range(n_blocks):
            # Only pass res_scale if it's provided
            block_kwargs = {
                'dim': dim,
                'kernel_size': kernel_size,
                'split_ratio': split_ratio,
                'norm_groups': norm_groups,
                'use_ea': use_ea,
            }
            if res_scale is not None:
                block_kwargs['res_scale'] = res_scale

            layers.append(PLKBlock(**block_kwargs))

        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Final convolution
        layers.append(nn.Conv2d(dim, out_ch * upscaling_factor**2, 3, 1, 1))

        self.feats = nn.Sequential(*layers)

        trunc_normal_(self.feats[0].weight, std=0.02)
        trunc_normal_(self.feats[-1].weight, std=0.02)

        self.repeat_op = partial(
            torch.repeat_interleave, repeats=upscaling_factor**2, dim=1
        )

        if dysample and upscaling_factor != 1:
            groups = out_ch if upscaling_factor % 2 != 0 else 4
            self.to_img = DySample(
                in_ch * upscaling_factor**2,
                out_ch,
                upscaling_factor,
                groups=groups,
                end_convolution=True if upscaling_factor != 1 else False,
            )
        else:
            self.to_img = nn.PixelShuffle(upscaling_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feats(x) + self.repeat_op(x)
        if not self.dysample or (self.dysample and self.upscale != 1):
            x = self.to_img(x)
        return x


def realplksr_s(**kwargs):
    return realplksr(n_blocks=12, kernel_size=13, use_ea=False, **kwargs)


def realplksr_l(**kwargs):
    return realplksr(dim=96, **kwargs)
