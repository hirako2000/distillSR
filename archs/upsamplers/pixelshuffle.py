"""
Standard PixelShuffle upsampler
"""

from torch import nn

from ..base import register_upsampler


@register_upsampler('pixelshuffle')
class PixelShuffleUpsampler(nn.Module):
    """Standard PixelShuffle upsampler"""

    def __init__(self, scale: int, **kwargs):
        super().__init__()
        self.scale = scale
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.pixel_shuffle(x)
