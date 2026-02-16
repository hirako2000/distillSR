"""
Model factory for CLI and config-based instantiation
"""

from .base import get_model


def create_model(model_name: str, **kwargs):
    """
    Create model by name with given parameters
    This is what the CLI will call
    """
    return get_model(model_name, **kwargs)


# Keep backward compatibility
def realplksr(**kwargs):
    return create_model('realplksr', **kwargs)


def realplksr_s(**kwargs):
    return create_model('realplksr_s', **kwargs)


def realplksr_l(**kwargs):
    return create_model('realplksr_l', **kwargs)
