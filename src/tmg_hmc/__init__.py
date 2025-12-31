__version__ = "0.1.3"

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

class _TensorPlaceholder(object):
        pass

def get_torch():
    if not _TORCH_AVAILABLE:
        return None
    return torch

def get_tensor_type():
    if not _TORCH_AVAILABLE:
        return _TensorPlaceholder
    return torch.Tensor

from .sampler import TMGSampler