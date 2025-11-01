__version__ = "0.1.0"

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .sampler import TMGSampler