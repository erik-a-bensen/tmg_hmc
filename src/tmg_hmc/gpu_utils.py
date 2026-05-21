import importlib.util

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if _TORCH_AVAILABLE:
    import torch

    Tensor = torch.Tensor
else:

    class Tensor:
        pass

    class torch:
        Tensor = Tensor
        sparse_coo = None
