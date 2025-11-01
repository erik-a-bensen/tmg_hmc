import numpy as np 
from cmath import acos
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix    
from typing import Tuple, Union
import os

from tmg_hmc import _TORCH_AVAILABLE
if _TORCH_AVAILABLE:
    from torch import Tensor, sparse_coo
else:
    class Tensor:
        pass

# ignore runtime warning
np.seterr(divide='ignore', invalid='ignore')

Array = Union[np.ndarray, Tensor, coo_matrix, None]
Sparse = Union[csc_matrix, csr_matrix, coo_matrix]
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

def to_scalar(x: Union[Array, float]) -> float:
    """
    Converts a scalar array or a float to a float.

    Parameters
    ----------
    x : Union[Array, float]
        The input value to be converted.

    Returns
    -------
    float
        The converted float value.
    """
    if isinstance(x, float):
        return x
    elif isinstance(x, Tensor):
        return x.item()
    elif len(x.shape) == 1:
        return x[0]
    return x[0,0]

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

    Notes
    -----
    Uses the cmath.acos function to handle complex values and returns the real part.
    Can potentially create ghost values if the input is outside the range [-1, 1]. 
    However, due to the complexity of the solution expressions this is necessary for 
    numerical stability and ghost solutions are filtered out later.
    """
    val = acos(x)
    return val.real