from hypothesis import given, strategies as st
import pytest
import numpy as np

from tmg_hmc.constraints import LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint
from tmg_hmc import _TORCH_AVAILABLE
if _TORCH_AVAILABLE:
    import torch
    from torch import Tensor
    GPU_AVAILABLE = torch.cuda.is_available()
else:
    torch = None
    GPU_AVAILABLE = False


@st.composite
def same_len_lists(draw, num_lists=2):
    # Draw a random length for all lists
    length = draw(st.integers(min_value=1, max_value=10)) 

    # Generate the specified number of lists with the drawn length
    lists = [
        draw(st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=length, max_size=length))
        for _ in range(num_lists)
    ]
    
    return tuple(lists)

@given(same_len_lists(), st.floats(min_value=-1e6, max_value=1e6))
def test_linear_constraint_value(input_lists, c):
    f, x = input_lists
    f = np.array(f).reshape(-1,1)
    x = np.array(x).reshape(-1,1)
    constraint = LinearConstraint(f, c)
    val = constraint.value(x)
    expected_val = f.T @ x + c
    assert isinstance(val, float)
    assert np.isclose(val, expected_val)

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(same_len_lists(), st.floats(min_value=-1e6, max_value=1e6))
def test_linear_constraint_value_gpu(input_lists, c):
    f, x = input_lists
    f = torch.tensor(f, device='cuda').reshape(-1,1)
    x = torch.tensor(x, device='cuda').reshape(-1,1)
    constraint = LinearConstraint(f, c)
    val = constraint.value(x)
    expected_val = f.T @ x + c
    assert isinstance(val, float)
    assert np.isclose(val, expected_val.cpu().item())

@given(same_len_lists(), st.floats(min_value=-1e6, max_value=1e6))
def test_linear_constraint_normal(input_lists, c):
    f, x = input_lists
    f = np.array(f).reshape(-1,1)
    x = np.array(x).reshape(-1,1)
    constraint = LinearConstraint(f, c)
    val = constraint.normal(x)
    expected_val = f
    assert isinstance(val, np.ndarray)
    assert np.isclose(val, expected_val).all()

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(same_len_lists(), st.floats(min_value=-1e6, max_value=1e6))
def test_linear_constraint_normal_gpu(input_lists, c):
    f, x = input_lists
    f = torch.tensor(f, device='cuda').reshape(-1,1)
    x = torch.tensor(x, device='cuda').reshape(-1,1)
    constraint = LinearConstraint(f, c)
    val = constraint.normal(x)
    expected_val = f
    assert isinstance(val, torch.Tensor)
    assert torch.isclose(val, expected_val).all()

@given(same_len_lists(num_lists=3), st.floats(min_value=-1e6, max_value=1e6))
def test_linear_constraint_hit_time(input_lists, c):
    f, x, v = input_lists
    f = np.array(f).reshape(-1,1)
    x = np.array(x).reshape(-1,1)
    v = np.array(v).reshape(-1,1)
    constraint = LinearConstraint(f, c)
    ts = constraint.hit_time(x, v)
    assert isinstance(ts, np.ndarray)
    assert len(ts) >= 1
    nans = np.isnan(ts)
    non_nan = ts[~nans]
    if len(non_nan) > 0:
        assert np.all(non_nan > 0)

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(same_len_lists(num_lists=3), st.floats(min_value=-1e6, max_value=1e6))
def test_linear_constraint_hit_time_gpu(input_lists, c):
    f, x, v = input_lists
    f = torch.tensor(f, device='cuda').reshape(-1,1)
    x = torch.tensor(x, device='cuda').reshape(-1,1)
    v = torch.tensor(v, device='cuda').reshape(-1,1)
    constraint = LinearConstraint(f, c)
    ts = constraint.hit_time(x, v)
    assert isinstance(ts, np.ndarray)
    assert len(ts) >= 1
    nans = np.isnan(ts)
    non_nan = ts[~nans]
    if len(non_nan) > 0:
        assert np.all(non_nan > 0)