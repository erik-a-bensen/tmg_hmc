from hypothesis import given, strategies as st
import pytest
import numpy as np

from tmg_hmc.constraints import LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint, ProductConstraint
from tmg_hmc import _TORCH_AVAILABLE
if _TORCH_AVAILABLE:
    import torch
    from torch import Tensor
    GPU_AVAILABLE = torch.cuda.is_available()
else:
    torch = None
    GPU_AVAILABLE = False

def equal_sets(a, b):
        if len(a) != len(b):
            return False
        for item in a:
            if not any(np.isclose(item, other, atol=1e-7, equal_nan=True) for other in b):
                return False
        return True

def test_initialize_product_constraint():
    constraint = LinearConstraint(f=np.array([[1.0], [2.0]]),c=0.0)
    constraint_list = [constraint] * 3
    product_constraint = ProductConstraint(constraint_list)
    assert len(product_constraint.constraints) == 3

@given(st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=2))
def test_value_product_constraint(x):
    x = np.array(x).reshape(-1, 1)
    constraint1 = LinearConstraint(f=np.array([[1.0], [0.0]]), c=-1.0)  # x[0] - 1 >= 0
    constraint2 = LinearConstraint(f=np.array([[0.0], [1.0]]), c=-2.0)  # x[1] - 2 >= 0
    product_constraint = ProductConstraint([constraint1, constraint2])
    expected_value = (x[0] - 1.0) * (x[1] - 2.0)
    result = product_constraint.value(x)
    assert np.isclose(result, expected_value, atol=1e-7)

@given(st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=2))
def test_normal_product_constraint(x):
    x = np.array(x).reshape(-1, 1)
    constraint1 = LinearConstraint(f=np.array([[1.0], [0.0]]), c=-1.0)  # x[0] - 1 >= 0
    constraint2 = LinearConstraint(f=np.array([[0.0], [1.0]]), c=-2.0)  # x[1] - 2 >= 0
    product_constraint = ProductConstraint([constraint1, constraint2])
    grad1 = constraint1.normal(x)
    grad2 = constraint2.normal(x)
    expected_normal = grad1 * (constraint2.value(x)) + grad2 * (constraint1.value(x))
    result_normal = product_constraint.normal(x)
    assert np.allclose(result_normal, expected_normal, atol=1e-7)

@given(st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=2), st.lists(st.floats(-1e6, 1e6), min_size=2, max_size=2))
def test_hit_time_product_constraint(x, v):
    x = np.array(x).reshape(-1, 1)
    v = np.array(v).reshape(-1, 1)
    constraint1 = LinearConstraint(f=np.array([[1.0], [0.0]]), c=-1.0)  # x[0] - 1 >= 0
    constraint2 = LinearConstraint(f=np.array([[0.0], [1.0]]), c=-2.0)  # x[1] - 2 >= 0
    product_constraint = ProductConstraint([constraint1, constraint2])
    
    t1s = constraint1.hit_time(x, v)
    t2s = constraint2.hit_time(x, v)
    
    prodts = product_constraint.hit_time(x, v)
    
    expected = set(t1s.tolist() + t2s.tolist())
    result = set(prodts.tolist())
    assert equal_sets(expected, result)