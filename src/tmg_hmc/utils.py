import numpy as np 
from cmath import acos
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from torch import Tensor, sparse_coo
from typing import TypeAlias, Tuple
import platform
import ctypes
import os
# ignore runtime warning
np.seterr(divide='ignore', invalid='ignore')

Array: TypeAlias = np.ndarray | Tensor | coo_matrix | None
Sparse: TypeAlias = csc_matrix | csr_matrix | coo_matrix 
base_path = os.path.dirname(os.path.abspath(__file__))

def get_shared_library() -> ctypes.CDLL:
    """
    Loads the compiled shared library for calculating the quadratic constraint hit times.

    Returns
    -------
        ctypes.CDLL: The loaded shared library.

    Notes
    -----
    The shared library is expected to be located at 'compiled/calc_solutions.{ext}'
    relative to the base path of this module, where {ext} is:
    - 'so' on Linux
    - 'dylib' on macOS
    - 'dll' on Windows

    Shared Library Function:
    - calc_all_solutions: Calculates all solutions for the quadratic constraint hit times.
        - Arguments: Five double precision floating-point numbers.
        - Returns: A pointer to an array of double precision floating-point numbers.

    Raises
    ------
    FileNotFoundError
        If the shared library is not found at the expected location.
    OSError
        If the operating system is unsupported.
    """
    # Determine shared library extension based on OS
    system = platform.system()
    if system == 'Linux':
        lib_ext = 'so'
    elif system == 'Darwin':  # macOS
        lib_ext = 'dylib'
    elif system == 'Windows':
        lib_ext = 'dll'
    else:
        raise OSError(f"Unsupported operating system: {system}")
    
    lib_path = os.path.join(base_path, 'compiled', f'calc_solutions.{lib_ext}')
    
    if not os.path.exists(lib_path):
        raise FileNotFoundError(
            f"Shared library not found at {lib_path}. "
            f"Please ensure the library has been compiled for your platform. "
            f"Run 'make' in the {os.path.join(base_path, 'compiled')} directory."
        )
    
    lib = ctypes.CDLL(lib_path)

    # Define function arguments
    lib.calc_all_solutions.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.calc_all_solutions.restype = ctypes.POINTER(ctypes.c_double)
    return lib

def check_installation() -> bool:
    """
    Checks if the shared library for calculating quadratic constraint hit times is installed.

    Raises
    ------
    FileNotFoundError
        If the shared library is not found at the expected location.
    OSError
        If the operating system is unsupported.
    """
    try:
        _ = get_shared_library()
        print("Installation check passed: Shared library is available.")
        return True
    except (FileNotFoundError, OSError) as e:
        print(f"Installation check failed: {e}")
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