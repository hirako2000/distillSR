"""
Upsampler factory for creating upsamplers by name
"""

from ..base import get_upsampler, register_upsampler


def create_upsampler(name: str, *args, **kwargs):
    """
    Create upsampler by name with given parameters
    
    Args:
        name: Upsampler name ('dysample', 'pixelshuffle')
        *args: Positional arguments to pass to the upsampler constructor
        **kwargs: Keyword arguments to pass to the upsampler constructor
    
    Returns:
        Upsampler module
    """
    return get_upsampler(name, *args, **kwargs)


__all__ = ['create_upsampler', 'register_upsampler']
