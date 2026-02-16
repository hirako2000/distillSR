"""
Base classes and registry for model architectures
"""

_MODEL_REGISTRY = {}
_BLOCK_REGISTRY = {}
_UPSAMPLER_REGISTRY = {}

def register_model(name):
    """Decorator to register model architectures"""
    def decorator(cls):
        _MODEL_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def register_block(name):
    """Decorator to register block types"""
    def decorator(cls):
        _BLOCK_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def register_upsampler(name):
    """Decorator to register upsamplers"""
    def decorator(cls):
        _UPSAMPLER_REGISTRY[name.lower()] = cls
        return cls
    return decorator

def get_model(name, **kwargs):
    """Get model by name with kwargs"""
    name = name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)

def get_block(name, **kwargs):
    """Get block by name"""
    name = name.lower()
    if name not in _BLOCK_REGISTRY:
        raise ValueError(f"Unknown block: {name}")
    return _BLOCK_REGISTRY[name](**kwargs)

def get_upsampler(name, *args, **kwargs):
    """Get upsampler by name"""
    name = name.lower()
    if name not in _UPSAMPLER_REGISTRY:
        raise ValueError(f"Unknown upsampler: {name}")
    return _UPSAMPLER_REGISTRY[name](*args, **kwargs)
