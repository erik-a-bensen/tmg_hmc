import numpy as np 
from cmath import sqrt, acos
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from torch import Tensor, sparse_coo
from typing import TypeAlias, Tuple
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
    The shared library is expected to be located at 'compiled/calc_solutions.so'
    relative to the base path of this module.

    Shared Library Function:
    - calc_all_solutions: Calculates all solutions for the quadratic constraint hit times.
        - Arguments: Five double precision floating-point numbers.
        - Returns: A pointer to an array of double precision floating-point numbers.
    """
    lib_path = os.path.join(base_path, 'compiled', 'calc_solutions.so')
    lib = ctypes.CDLL(lib_path)

    # Define function arguments
    lib.calc_all_solutions.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double]
    lib.calc_all_solutions.restype = ctypes.POINTER(ctypes.c_double)
    return lib

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

def soln1(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the first of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) - 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln2(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the second of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) - 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln3(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the third of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) - 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln4(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the fourth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) - 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln5(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the fifth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) + 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln6(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the sixth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) + 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln7(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the seventh of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) + 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln8(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
    Computes the eighth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) + 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))
