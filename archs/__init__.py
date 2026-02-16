"""
Architecture modules for RealPLKSR Factory
"""

from . import upsamplers  # noqa: F401
from .blocks.common import DCCM, EA
from .blocks.conv import PLKConv2d
from .blocks.plk_block import PLKBlock
from .factory import create_model
from .realplksr import realplksr, realplksr_l, realplksr_s
from .upsamplers import DySample, PixelShuffleUpsampler, create_upsampler

__all__ = [
    'realplksr',
    'realplksr_s',
    'realplksr_l',
    'create_model',
    'DCCM',
    'EA',
    'PLKConv2d',
    'PLKBlock',
    'DySample',
    'PixelShuffleUpsampler',
    'create_upsampler',
]
