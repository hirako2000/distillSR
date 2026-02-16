"""
Architecture modules for RealPLKSR Factory
"""

# Import upsamplers package FIRST to ensure decorators run
from . import upsamplers

# Export components for testing
from .blocks.common import DCCM, EA
from .blocks.conv import PLKConv2d  # Remove PLKConv2dV2
from .blocks.plk_block import PLKBlock
from .factory import create_model

# Export main model classes for backward compatibility
from .realplksr import realplksr, realplksr_l, realplksr_s

# Export upsamplers
from .upsamplers import DySample, PixelShuffleUpsampler, create_upsampler

__all__ = [
    'realplksr',
    'realplksr_s',
    'realplksr_l',
    'create_model',
    'DCCM',
    'EA',
    'PLKConv2d',  # Remove PLKConv2dV2
    'PLKBlock',
    'DySample',
    'PixelShuffleUpsampler',
    'create_upsampler',
]
