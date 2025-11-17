from hypothesis import given, assume, strategies as st
from hypothesis.extra.numpy import arrays
import pytest
import os
import numpy as np
import scipy.sparse as sp

from tmg_hmc import TMGSampler, _TORCH_AVAILABLE
from tmg_hmc.constraints import LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint
if _TORCH_AVAILABLE:
    import torch
    gpu_available = torch.cuda.is_available()
else:
    torch = None
    gpu_available = False

@st.composite
def positive_definite_matrices(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate a random matrix
    A = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e-6, 1e-6)))
    # Make it symmetric positive definite
    pd_matrix = A.T @ A + np.eye(size) * 1e-6  # Add small value to diagonal for numerical stability
    return pd_matrix

@st.composite
def square_matrix(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    mat = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e6, 1e6)))
    return mat

@st.composite
def symmetric_matrix(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    mat = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e6, 1e6)))
    sym_mat = (mat + mat.T) / 2
    return sym_mat

@st.composite
def vector(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    vec = draw(arrays(dtype=np.float64, shape=(size, 1), elements=st.floats(-1e6, 1e6)))
    return vec

@st.composite
def sym_mat_vec(draw, min_size=2, max_size=10):
    size = draw(st.integers(min_value=min_size, max_value=max_size))
    mat = draw(arrays(dtype=np.float64, shape=(size, size), elements=st.floats(-1e6, 1e6)))
    sym_mat = (mat + mat.T) / 2
    vec = draw(arrays(dtype=np.float64, shape=(size, 1), elements=st.floats(-1e6, 1e6)))
    return sym_mat, vec

@given(positive_definite_matrices())
def test_sampler_initialization_errors(pd_matrix):
    with pytest.raises(ValueError):
        TMGSampler() 
    TMGSampler(Sigma=pd_matrix)  # Should not raise
    TMGSampler(Sigma_half=pd_matrix)  # Should not raise

    with pytest.raises(ValueError):
        TMGSampler(Sigma=pd_matrix[:-1, :])  # Invalid shape
    with pytest.raises(ValueError):
        TMGSampler(Sigma_half=pd_matrix[:-1, :])  # Invalid shape
    
    asym_matrix = pd_matrix.copy()
    asym_matrix[0,1] += 1e-3  # Make it asymmetric
    with pytest.raises(ValueError):
        TMGSampler(Sigma=asym_matrix)  # Not symmetric
    with pytest.raises(ValueError):
        TMGSampler(Sigma_half=asym_matrix)  # Not symmetric

    evals = np.linalg.eigvalsh(pd_matrix)
    npsd_matrix = pd_matrix - 2 * np.min(evals) * np.eye(pd_matrix.shape[0]) - 1e-3 * np.eye(pd_matrix.shape[0])  # Make it not positive semi-definite
    with pytest.raises(ValueError):
        TMGSampler(Sigma=npsd_matrix)  # Not positive semi-definite
    with pytest.raises(ValueError):
        TMGSampler(Sigma_half=npsd_matrix)  # Not positive semi-definite

@given(positive_definite_matrices())
def test_sigma_decomposition(pd_matrix):
    sampler = TMGSampler(Sigma=pd_matrix)
    Sigma_half = sampler.Sigma_half
    reconstructed_Sigma = Sigma_half @ Sigma_half.T
    assert np.allclose(reconstructed_Sigma, pd_matrix)

@given(st.integers(min_value=2, max_value=10))
def test_invalid_constraint_errors(dim):
    sampler = TMGSampler(Sigma=np.eye(dim))
    with pytest.raises(ValueError):
        sampler.add_constraint() # Neither A nor f provided
    with pytest.raises(ValueError):
        A = np.zeros((dim, dim))
        f = np.zeros((dim, 1))
        sampler.add_constraint(A=A, f=f)  # Both A and f zero

@given(vector())
def test_add_linear_constraint(f):
    assume(float(np.linalg.norm(f)) > 1e-8)  # Avoid zero vector
    sampler = TMGSampler(Sigma=np.eye(f.shape[0]))
    sampler.add_constraint(f=f)
    assert len(sampler.constraints) == 1
    constraint = sampler.constraints[0]
    assert isinstance(constraint, LinearConstraint)
    assert np.allclose(constraint.f, f)

@given(symmetric_matrix())
def test_add_simple_quadratic_constraint(A):
    assume(not np.allclose(A, 0))  # Avoid zero matrix
    sampler = TMGSampler(Sigma=np.eye(A.shape[0]))
    sampler.add_constraint(A=A, sparse=False)
    assert len(sampler.constraints) == 1
    constraint = sampler.constraints[0]
    assert isinstance(constraint, SimpleQuadraticConstraint)
    assert np.allclose(constraint.A, A)

@given(sym_mat_vec())
def test_add_quadratic_constraint(data):
    A, f = data
    assume(not np.allclose(A, 0) and not np.allclose(f, 0))  # Avoid zero matrix and zero vector
    sampler = TMGSampler(Sigma=np.eye(A.shape[0]))
    sampler.add_constraint(A=A, f=f, sparse=False)
    assert len(sampler.constraints) == 1
    constraint = sampler.constraints[0]
    assert isinstance(constraint, QuadraticConstraint)
    assert np.allclose(constraint.A, A)
    assert np.allclose(constraint.b, f)

@given(st.integers(min_value=2, max_value=10))
def test_unconstrained_sampling(dim):
    sampler = TMGSampler(Sigma=np.eye(dim))
    np.random.seed(0)
    sample = sampler.sample(x0=np.zeros((dim,1)), n_samples=5, burn_in=0)
    np.random.seed(0)
    expected_samples = np.random.multivariate_normal(mean=np.zeros(dim), cov=np.eye(dim), size=5)
    assert np.allclose(sample, expected_samples)

@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available, reason="GPU not available")
@given(st.integers(min_value=2, max_value=10))
def test_unconstrained_sampling_gpu(dim):
    sampler = TMGSampler(Sigma=np.eye(dim), gpu=True)
    torch.manual_seed(0)
    sample = sampler.sample(x0=np.zeros((dim,1)), n_samples=5, burn_in=0)
    torch.manual_seed(0)
    expected_samples = []
    for _ in range(5):
        expected_samples.append(torch.randn(dim, 1, dtype=torch.float64).cpu().numpy())
    expected_samples = np.concatenate(expected_samples, axis=1).T

    assert np.allclose(sample, expected_samples)

def test_save_load():
    sampler = TMGSampler(Sigma=np.eye(3))
    sampler.add_constraint(f=np.array([[1.0], [0.0], [0.0]]), sparse=False)
    sampler.add_constraint(A=np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]), sparse=False)
    sampler.add_constraint(A=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), 
                           f=np.array([[1.0], [1.0], [1.0]]), sparse=False)

    sampler.save("test_sampler.pkl")
    loaded_sampler = TMGSampler.load("test_sampler.pkl")

    assert np.allclose(loaded_sampler.Sigma_half, sampler.Sigma_half)
    assert len(loaded_sampler.constraints) == len(sampler.constraints)
    for c1, c2 in zip(loaded_sampler.constraints, sampler.constraints):
        assert type(c1) == type(c2)
        if isinstance(c1, LinearConstraint):
            assert np.allclose(c1.f, c2.f)
        elif isinstance(c1, SimpleQuadraticConstraint):
            assert np.allclose(c1.A, c2.A)
        elif isinstance(c1, QuadraticConstraint):
            assert np.allclose(c1.A, c2.A)
            assert np.allclose(c1.b, c2.b)
    os.remove("test_sampler.pkl")