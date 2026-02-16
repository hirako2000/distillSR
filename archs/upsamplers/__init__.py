"""
Upsampler modules for super-resolution
"""

# Import these first so decorators run IMMEDIATELY when module loads
from .dysample import DySample

# Now import factory (which uses the registry)
from .factory import create_upsampler, register_upsampler
from .pixelshuffle import PixelShuffleUpsampler

__all__ = [
    'DySample',
    'PixelShuffleUpsampler',
    'create_upsampler',
    'register_upsampler',
]
