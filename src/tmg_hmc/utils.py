import numpy as np 
import cmath
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix    
from typing import Tuple, TypeAlias
import os
from tmg_hmc import get_torch, get_tensor_type

torch, Tensor = get_torch(), get_tensor_type()

# ignore runtime warning
np.seterr(divide='ignore', invalid='ignore')

Array: TypeAlias = np.ndarray | Tensor | coo_matrix | None
Sparse: TypeAlias = csc_matrix | csr_matrix | coo_matrix
base_path = os.path.dirname(os.path.abspath(__file__))

def compiled_library_available() -> bool:
    """
    Checks if the compiled shared library is available.

    Return
    -------
    bool
        True if the shared library is available, False otherwise.
    """
    try:
        import tmg_hmc.compiled as c 
        return True
    except ImportError:
        return False

def sparsify(A: Array) -> Array:
    """
    Converts a dense numpy array or a PyTorch tensor to a sparse COO matrix.

    Parameters
    ----------
    A : Array
        The input array to be converted to a sparse matrix.

    Returns
    -------
    Array
        The sparse COO matrix representation of the input array.
    """
    if isinstance(A, np.ndarray):
        return coo_matrix(A)
    elif isinstance(A, Tensor):
        return A.to_sparse()
    else:
        raise ValueError(f"Unknown type {type(A)}")

def get_sparse_elements(A: Array) -> Tuple[Array, Array, Array]:
    """
    Extracts the row, column, and data elements from a sparse matrix.

    Parameters
    ----------
    A : Array
        The input sparse matrix.

    Returns
    -------
    Tuple[Array, Array, Array]
        A tuple containing the row indices, column indices, and data values of the sparse matrix.
    """
    if isinstance(A, coo_matrix):
        return A.row, A.col, A.data
    elif isinstance(A, Tensor):
        if A.layout == sparse_coo:
            row, col = A.indices()
            return row, col, A.values()
        else:
            row, col = A.nonzero().unbind(1)
            return row, col, A[row, col]
    elif isinstance(A, np.ndarray):
        row, col = np.nonzero(A)
        return row, col, A[row, col]
    else:
        raise ValueError(f"Unknown type {type(A)}")

def to_scalar(x: Array | float) -> float:
    """
    Converts a scalar array or a float to a float.

    Parameters
    ----------
    x : Array | float
        The input value to be converted.

    Returns
    -------
    float
        The converted float value.
    """
    if isinstance(x, float):
        return x
    elif isinstance(x, Tensor):
        return x.cpu().item()
    elif len(x.shape) == 1:
        return x[0]
    return x[0,0]

def is_nonzero_array(x: Array) -> bool:
    """
    Checks if the input array is non-zero.

    Parameters
    ----------
    x : Array
        The input array to be checked.

    Returns
    -------
    bool
        True if the array is non-zero, False otherwise.
    """
    if isinstance(x, Tensor):
        return not torch.all(x == 0).item()
    elif isinstance(x, np.ndarray):
        return not np.allclose(x, 0)
    elif isinstance(x, coo_matrix):
        return x.nnz > 0
    else:
        raise ValueError(f"Unknown type {type(x)}")

def stable_acos(x: complex) -> complex:
    """
    Computes a numerically stable arccosine for complex numbers.

    Parameters
    ----------
    x : complex
        The input complex number.

    Returns
    -------
    complex
        The arccosine of the input complex number.
    """
    if np.abs(x) > 1:
        return cmath.acos(x)
    else:
        return -1j * cmath.log(x + 1j*cmath.sqrt(1 - x*x))

def arccos(x: float) -> float:
    """
    Computes the real component of the arccosine of a value.

    Parameters
    ----------
    x : float
        The input value.

    Returns
    -------
    float
        The real component of the arccosine of the input value.
    """
    val = stable_acos(x)
    if abs(val.imag) > 1e-1:
        return np.nan  # Return NaN for significant imaginary parts
    return val.real
