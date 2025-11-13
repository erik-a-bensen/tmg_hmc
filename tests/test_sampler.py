from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays
import pytest
import numpy as np
import scipy.sparse as sp

from tmg_hmc import TMGSampler, _TORCH_AVAILABLE
if _TORCH_AVAILABLE:
    import torch
    gpu_available = torch.cuda.is_available()
else:
    torch = None
    gpu_available = False
