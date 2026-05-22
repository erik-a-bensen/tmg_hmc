import importlib.util
from typing import TYPE_CHECKING

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TYPE_CHECKING:
    import torch
    from torch import Tensor
elif _TORCH_AVAILABLE:
    import torch
    Tensor = torch.Tensor
else:
    class Tensor:  # type: ignore[no-redef]
        pass

    class torch:  # type: ignore[no-redef]
        Tensor = Tensor
        sparse_coo = None