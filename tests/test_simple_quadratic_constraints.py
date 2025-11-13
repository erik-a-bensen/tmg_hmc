from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays
import pytest
import numpy as np
import scipy.sparse as sp

from tmg_hmc.constraints import SimpleQuadraticConstraint
from tmg_hmc import _TORCH_AVAILABLE
if _TORCH_AVAILABLE:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
else:
    torch = None
    GPU_AVAILABLE = False


@st.composite
def A_S_vecs(draw, num_vecs=1):
    # Draw a random length for all lists
    length = draw(st.integers(min_value=1, max_value=10)) 

    # Generate matrix of shape length x length
    Amatrix = draw(arrays(dtype=np.float64, shape=(length, length), elements=st.floats(min_value=-1e6, max_value=1e6)))
    Amatrix = (Amatrix + Amatrix.T) / 2  # Ensure the matrix is symmetric

    # Generate PSD S matrix
    Smatrix = draw(arrays(dtype=np.float64, shape=(length, length), elements=st.floats(min_value=-1e6, max_value=1e6)))
    Smatrix = Smatrix @ Smatrix.T  # Make it positive semi-definite
    Smatrix += np.eye(length) * 1e-6  # Add small value to diagonal for numerical stability

    # Generate the specified number of vectors with the drawn length
    vectors = [
        draw(arrays(dtype=np.float64, shape=(length, 1), elements=st.floats(min_value=-1e6, max_value=1e6)))
        for _ in range(num_vecs)
    ]
    
    return (Amatrix, Smatrix) + tuple(vectors)

@st.composite
def Asparse_S_vecs(draw, num_vecs=1):
    # Draw a random length for all lists
    length = draw(st.integers(min_value=2, max_value=10)) 

    # Generate sparse matrix of shape length x length
    density = draw(st.floats(min_value=0.01, max_value=0.5))
    Amatrix = sp.random(length, length, density=density, format='coo', dtype=np.float64)
    Amatrix = (Amatrix + Amatrix.T) / 2  # Ensure the matrix is symmetric
    Amatrix = Amatrix.toarray()

    # Generate PSD S matrix
    Smatrix = draw(arrays(dtype=np.float64, shape=(length, length), elements=st.floats(min_value=-1e6, max_value=1e6)))
    Smatrix = Smatrix @ Smatrix.T  # Make it positive semi-definite
    Smatrix += np.eye(length) * 1e-6  # Add small value to diagonal for numerical stability

    # Generate the specified number of vectors with the drawn length
    vectors = [
        draw(arrays(dtype=np.float64, shape=(length, 1), elements=st.floats(min_value=-1e6, max_value=1e6)))
        for _ in range(num_vecs)
    ]
    
    return (Amatrix, Smatrix) + tuple(vectors)

@given(A_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_value(input_lists, c):
    A, S, x = input_lists
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=False)
    val = constraint.value(x)
    Atilde = S @ A @ S
    assert np.allclose(constraint.A, Atilde)
    expected_val = x.T @ Atilde @ x + c
    assert isinstance(val, float)
    assert np.isclose(val, expected_val)

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(A_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_value_gpu(input_lists, c):
    A, S, x = input_lists
    A = torch.tensor(A, device='cuda')
    S = torch.tensor(S, device='cuda')
    x = torch.tensor(x, device='cuda')
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=False)
    val = constraint.value(x)
    Atilde = S @ A @ S
    assert torch.allclose(constraint.A, Atilde)
    expected_val = x.T @ Atilde @ x + c
    assert isinstance(val, float)
    assert np.isclose(val, expected_val.cpu().item())

@given(Asparse_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_value_sparse(input_lists, c):
    A, S, x = input_lists
    assume(not np.allclose(A, np.zeros_like(A)))  # Ensure A is not the zero matrix
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=True)
    val = constraint.value(x)
    Atilde = S @ A @ S # Note: do not test Atilde, not stored in sparse case
    expected_val = x.T @ Atilde @ x + c
    assert isinstance(val, float)
    assert np.isclose(val, expected_val)

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(Asparse_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_value_sparse_gpu(input_lists, c):
    A, S, x = input_lists
    assume(not np.allclose(A, np.zeros_like(A)))  # Ensure A is not the zero matrix
    A = torch.tensor(A, device='cuda')
    S = torch.tensor(S, device='cuda')
    x = torch.tensor(x, device='cuda')
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=True)
    val = constraint.value(x)
    Atilde = S @ A @ S # Note: do not test Atilde, not stored in sparse case
    expected_val = x.T @ Atilde @ x + c
    assert isinstance(val, float)
    assert np.isclose(val, expected_val.cpu().item())

@given(A_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_normal(input_lists, c):
    A, S, x = input_lists
    Atilde = S @ A @ S
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=False)
    val = constraint.normal(x)
    expected_val = 2 * Atilde @ x
    assert isinstance(val, np.ndarray)
    assert np.isclose(val, expected_val).all()

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(A_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_normal_gpu(input_lists, c):
    A, S, x = input_lists
    A = torch.tensor(A, device='cuda')
    S = torch.tensor(S, device='cuda')
    x = torch.tensor(x, device='cuda')
    Atilde = S @ A @ S
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=False)
    val = constraint.normal(x)
    expected_val = 2 * Atilde @ x
    assert isinstance(val, torch.Tensor)
    assert torch.isclose(val, expected_val).all()

@given(Asparse_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_normal_sparse(input_lists, c):
    A, S, x = input_lists
    assume(not np.allclose(A, np.zeros_like(A)))  # Ensure A is not the zero matrix
    Atilde = S @ A @ S
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=True)
    val = constraint.normal(x)
    expected_val = 2 * Atilde @ x
    assert isinstance(val, np.ndarray)
    assert np.isclose(val, expected_val).all()

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(Asparse_S_vecs(), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_normal_sparse_gpu(input_lists, c):
    A, S, x = input_lists
    assume(not np.allclose(A, np.zeros_like(A)))  # Ensure A is not the zero matrix
    A = torch.tensor(A, device='cuda')
    S = torch.tensor(S, device='cuda')
    x = torch.tensor(x, device='cuda')
    Atilde = S @ A @ S
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=True)
    val = constraint.normal(x)
    expected_val = 2 * Atilde @ x
    assert isinstance(val, torch.Tensor)
    assert torch.isclose(val, expected_val).all()

@given(A_S_vecs(num_vecs=2), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_hit_time(input_lists, c):
    A, S, x, v = input_lists
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=False)
    ts = constraint.hit_time(x, v)
    assert isinstance(ts, np.ndarray)
    nans = np.isnan(ts)
    non_nan = ts[~nans]
    if len(non_nan) > 0:
        assert np.all(non_nan > 0)

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(A_S_vecs(num_vecs=2), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_hit_time_gpu(input_lists, c):

    A, S, x, v = input_lists
    A = torch.tensor(A, device='cuda')
    S = torch.tensor(S, device='cuda')
    x = torch.tensor(x, device='cuda')
    v = torch.tensor(v, device='cuda')
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=False)
    ts = constraint.hit_time(x, v)
    assert isinstance(ts, np.ndarray)
    nans = np.isnan(ts)
    non_nan = ts[~nans]
    if len(non_nan) > 0:
        assert np.all(non_nan > 0)

@given(Asparse_S_vecs(num_vecs=2), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_hit_time_sparse(input_lists, c):
    A, S, x, v = input_lists
    assume(not np.allclose(A, np.zeros_like(A)))  # Ensure A is not the zero matrix
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=True)
    ts = constraint.hit_time(x, v)
    assert isinstance(ts, np.ndarray)
    nans = np.isnan(ts)
    non_nan = ts[~nans]
    if len(non_nan) > 0:
        assert np.all(non_nan > 0)

@pytest.mark.gpu
@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
@given(Asparse_S_vecs(num_vecs=2), st.floats(min_value=-1e6, max_value=1e6))
def test_simple_quadratic_constraint_hit_time_sparse_gpu(input_lists, c):
    A, S, x, v = input_lists
    assume(not np.allclose(A, np.zeros_like(A)))  # Ensure A is not the zero matrix
    A = torch.tensor(A, device='cuda')
    S = torch.tensor(S, device='cuda')
    x = torch.tensor(x, device='cuda')
    v = torch.tensor(v, device='cuda')
    constraint = SimpleQuadraticConstraint(A=A, c=c, S=S, sparse=True)
    ts = constraint.hit_time(x, v)
    assert isinstance(ts, np.ndarray)
    nans = np.isnan(ts)
    non_nan = ts[~nans]
    if len(non_nan) > 0:
        assert np.all(non_nan > 0)