__version__ = "0.0.22"

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from .sampler import TMGSampler