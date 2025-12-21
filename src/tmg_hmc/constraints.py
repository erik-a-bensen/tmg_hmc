from __future__ import annotations
import numpy as np
from typing import Protocol, Tuple, Protocol
from tmg_hmc.utils import Array, Sparse, to_scalar, get_sparse_elements
from tmg_hmc.quad_solns import soln1, soln2, soln3, soln4, soln5, soln6, soln7, soln8
from tmg_hmc import get_torch, get_tensor_type
from tmg_hmc.compiled import calc_all_solutions
import warnings

torch, Tensor = get_torch(), get_tensor_type()

pis = np.array([-1, 0, 1]) * np.pi
eps = 1e-12

class Constraint(Protocol):
    """
    Abstract base class for constraints
    """
    def value(self, x: Array) -> float:
        """
        Compute the value of the constraint at x
        """
        pass

    def is_satisfied(self, x: Array) -> bool:
        """
        Check if the constraint is satisfied at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise
        """
        return self.value(x) >= 0 

    def is_zero(self, x: Array) -> Tuple[bool, bool]:
        """
        Check if the constraint is zero at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        Tuple[bool, bool]
            (is_strictly_zero, is_approximately_zero)
        """
        val = self.value(x)
        return np.isclose(val, 0), np.isclose(val, 0, atol=1e-2)
    
    def compute_q(self, a: Array, b: Array) -> Tuple[float, ...]:
        """
        Compute the coefficients of the constraint equation along the trajectory defined by a and b
        """
        pass

    def hit_time(self, a: Array, b: Array) -> Array:
        """
        Compute the hit time of the constraint along the trajectory defined by a and b
        """
        pass

    def normal(self, x: Array) -> Array:
        """
        Compute the normal vector of the constraint at x
        """
        pass

    def reflect(self, x: Array, xdot: Array) -> Array:
        """
        Reflect the velocity xdot at the constraint surface defined by x

        Parameters
        ----------
        x : Array
            Point on the constraint surface
        xdot : Array
            Velocity to be reflected

        Returns
        -------
        Array
            Reflected velocity
        """
        f = self.normal(x)
        if isinstance(xdot, Tensor):
            norm = torch.sqrt(f.T @ f)
        else:
            norm = np.sqrt(f.T @ f)
        f = f / norm
        return xdot - 2 * (f.T @ xdot) * f

    def serialize(self) -> dict:
        """
        Serialize the constraint to a dictionary

        Returns
        -------
        dict
            Dictionary representation of the constraint
        """
        d = self.__dict__.copy()
        
        # For sparse constraints, ensure we save S directly
        # and remove the individual row/column vectors that cause reconstruction issues
        if 'sparse' in d and d['sparse']:
            # Keep S if it exists
            if 'S' not in d and hasattr(self, 'S'):
                d['S'] = self.S
            # Remove problematic sparse reconstruction data
            keys_to_remove = ['s_rows', 's_cols', 'row_data', 'col_data']
            for key in keys_to_remove:
                if key in d:
                    del d[key]
        
        # Convert tensors to CPU
        for k, v in d.items():
            if isinstance(v, Tensor):
                d[k] = v.cpu()
        
        d['type'] = self.__class__.__name__
        return d
    
    @classmethod
    def deserialize(cls, d: dict, gpu: bool) -> Constraint:
        """
        Deserialize the constraint from a dictionary

        Parameters
        ----------
        d : dict
            Dictionary representation of the constraint
        gpu : bool
            Whether to load tensors onto the GPU

        Returns
        -------
        Constraint
            Deserialized constraint object
        """
        if gpu:
            for k, v in d.items():
                if isinstance(v, Tensor):
                    d[k] = v.cuda()
        if d['type'] == 'LinearConstraint':
            return LinearConstraint(d['f'], d['c'])
        elif d['type'] == 'SimpleQuadraticConstraint':
            return SimpleQuadraticConstraint.build_from_dict(d, gpu)
        elif d['type'] == 'QuadraticConstraint':
            return QuadraticConstraint.build_from_dict(d, gpu)
        else:
            raise ValueError(f"Unknown constraint type {d['type']}")
    

class ProductConstraint(Constraint):
    """
    Constraint that is the product of multiple linear or quadratic constraints
    """
    def __init__(self, constraints: Tuple[Constraint, ...]) -> None:
        """
        Parameters
        ----------
        constraints : Tuple[Constraint, ...]
            Tuple of constraints to be combined
        """
        self.constraints = constraints

    def value(self, x: Array) -> float:
        """
        Compute the value of the product constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the product constraint at x
        """
        val = 1.0
        for constraint in self.constraints:
            val *= constraint.value(x)
        return val
    
    def normal(self, x: Array) -> Array:
        """
        Compute the normal vector of the product constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector of the product constraint at x
        """
        vals = [c.value(x) for c in self.constraints]
        normals = [c.normal(x) for c in self.constraints]
        weighted = [normals[i] * np.prod(vals[:i] + vals[i+1:]) for i in range(len(self.constraints))]
        return sum(weighted)
    
    def hit_time(self, x: Array, xdot: Array) -> Array:
        """
        Compute the hit time of the product constraint along the trajectory defined by x and xdot

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            Hit times of the product constraint along the trajectory
        """
        hit_times = []
        for constraint in self.constraints:
            ht = constraint.hit_time(x, xdot)
            hit_times.append(ht)
        return np.concatenate(hit_times)


class LinearConstraint(Constraint):
    """
    Constraint of the form fx + c >= 0
    """
    def __init__(self, f: Array, c: float) -> None:
        """ 
        Parameters
        ----------
        f : Array
            Coefficient vector
        c : float
            Constant term
        """
        self.f = f
        self.c = c
    
    def value(self, x: Array) -> float:
        """
        Compute the value of the constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the constraint at x given by f^T x + c
        """
        return to_scalar(self.f.T @ x + self.c)

    def normal(self, x: Array) -> Array:
        """
        Compute the normal vector of the constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector of the constraint at x given by f
        """
        return self.f

    def compute_q(self, a: Array, b: Array) -> Tuple[float, float]:
        """
        Compute the 2 q terms for the linear constraint

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float]
            q terms for the constraint

        Notes
        -----
        These expressions are defined such that Eqn 2.22 in Pakman and Paninski (2014) 
        simplifies to: q1 sin(t) + q2 cos(t) + c = 0
        """
        f = self.f
        q1 = to_scalar(f.T @ a)
        q2 = to_scalar(f.T @ b)
        return q1, q2

    def hit_time(self, x: Array, xdot: Array) -> Array:
        """
        Compute the hit time of the constraint along the trajectory defined by x and xdot

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            Hit time of the constraint along the trajectory

        Notes
        -----
        Hit time is computed by solving Eqn 2.26 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Due to the sum of inverse trig functions, we check the solution and 
        the solution +- pi to ensure we capture all hit times. 

        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.
        """
        q1, q2 = self.compute_q(xdot, x)
        c = self.c
        u = np.sqrt(q1**2 + q2**2)
        if (u < abs(c)) or (u == 0) or (q2 == 0): 
            # No intersection so return NaN
            return np.array([np.nan])
        s1 = -np.arccos(-c/u) + np.arctan(q1/q2) + pis
        s2 = np.arccos(-c/u) + np.arctan(q1/q2) + pis
        s = np.hstack([s1, s2])
        return s[s > eps]



class BaseQuadraticConstraint(Constraint):
    """
    Base class for quadratic constraints
    """
    def _setup_values(self, A: Array, S: Array) -> None:
        """
        Setup internal values for dense matrix computation
        
        Parameters
        ----------
        A : Array
            Quadratic coefficient matrix
        S : Array
            Transformation matrix given by the Symmetric Sqrt of the Mass matrix

        Notes
        -----
        Sets up the internal methods for value, normal, and compute_q to use 
        dense matrix computations.
        """
        self.A_orig = A
        self.S = S
        self.value = self.value_
        self.normal = self.normal_
        self.compute_q = self.compute_q_

    def _setup_values_sparse(self, A: Array, S: Array) -> None:
        """
        Setup internal values for sparse matrix computation

        Parameters
        ----------
        A : Array
            Quadratic coefficient matrix
        S : Array
            Transformation matrix given by the Symmetric Sqrt of the Mass matrix

        Notes
        -----
        Sets up the internal methods for value, normal, and compute_q to use
        sparse matrix computations.
        """
        rows, cols, vals = get_sparse_elements(A)
        self.n_comps = len(rows)
        self.n = A.shape[0]
        self.A_orig = A
        print(S)
        self.s_rows = [S[i,:].reshape((1,self.n)) for i in rows] # S[i,:] is a row vector
        self.s_cols = [S[:,j].reshape((self.n,1)) for j in cols] # S[:,j] is a column vector
        self.a_vals = vals.reshape((self.n_comps,))
        self.value = self.value_sparse
        self.normal = self.normal_sparse
        self.compute_q = self.compute_q_sparse

    def value_(self, x: Array) -> float:
        """Placeholder method for dense value computation"""
        pass

    def value_sparse(self, x: Array) -> float:
        """Placeholder method for sparse value computation"""
        pass

    def normal_(self, x: Array) -> Array:
        """Placeholder method for dense normal vector computation"""
        pass

    def normal_sparse(self, x: Array) -> Array:
        """Placeholder method for sparse normal vector computation"""
        pass

    def compute_q_(self, a: Array, b: Array) -> Tuple[float, ...]:
        """Placeholder method for dense q term computation"""
        pass

    def compute_q_sparse(self, a: Array, b: Array) -> Tuple[float, ...]:
        """Placeholder method for sparse q term computation"""
        pass

    @property 
    def A(self):
        """Compute the transformed quadratic matrix A = S A_orig S on the fly"""
        return self.S @ self.A_orig @ self.S
    
    def A_dot_x(self, x: Array) -> Array:
        """
        Compute A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate A x at

        Returns
        -------
        Array
            Result of A x computation
        """
        dot_prods = [self.s_rows[i].reshape((1,self.n)) @ x for i in range(self.n_comps)]
        return sum([self.a_vals[i]*dot_prods[i]*self.s_cols[i].reshape((self.n,1)) for i in range(self.n_comps)])

    def x_dot_A_dot_x(self, x: Array) -> float:
        """
        Compute x^T A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate x^T A x at

        Returns
        -------
        float
            Result of x^T A x computation
        """
        return x.T @ self.A_dot_x(x)

class SimpleQuadraticConstraint(BaseQuadraticConstraint):
    """
    Constraint of the form x^T A x + c >= 0
    """
    def __init__(self, A: Array, c: float, S: Array, sparse: bool = False):
        """
        Parameters
        ----------
        A : Array
            Quadratic coefficient matrix
        c : float
            Constant term
        S : Array
            Transformation matrix given by the Symmetric Sqrt of the Mass matrix
        sparse : bool, optional
            Whether to use sparse matrix computations, by default False

        Notes
        -----
        If A is a sparse matrix, sparse computations are used regardless of the
        sparse parameter.
        """
        self.c = c
        if isinstance(A, Sparse):
            sparse = True
        self.sparse = sparse
        if sparse:
            self._setup_values_sparse(A, S)
        else:
            self._setup_values(A, S)

    @classmethod 
    def build_from_dict(cls, d: dict, gpu: bool) -> SimpleQuadraticConstraint:
        """
        Build a SimpleQuadraticConstraint from a dictionary representation

        Parameters
        ----------
        d : dict
            Dictionary representation of the constraint
        gpu : bool
            Whether to load tensors onto the GPU

        Returns
        -------
        SimpleQuadraticConstraint
            The constructed constraint
        """
        if gpu and not _TORCH_AVAILABLE:
            gpu = False
            warnings.warn("GPU requested but PyTorch is not available. Loading on CPU instead.")
        sparse = d['sparse']
        A = d['A_orig']
        c = d['c']
        S = d.get('S', None)
        
        # Move to GPU if requested
        if gpu:
            if isinstance(S, Tensor):
                S = S.cuda()
            if isinstance(A, Tensor):
                A = A.cuda()
        
        return cls(A, c, S, sparse)
    
    def value_(self, x: Array) -> float:
        """
        Compute the value of the constraint at x using dense matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the constraint at x given by x^T A x + c
        """
        return to_scalar(x.T @ self.A @ x + self.c)
    
    def value_sparse(self, x: Array) -> float:
        """
        Compute the value of the constraint at x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the constraint at x given by x^T A x + c
        """
        return to_scalar(x.T @ self.A_dot_x(x) + self.c)

    def normal_(self, x: Array) -> Array:
        """
        Compute the normal vector at x using dense matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x
        """
        return 2 * self.A @ x
    
    def normal_sparse(self, x: Array) -> Array:
        """
        Compute the normal vector at x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x
        """
        return 2 * self.A_dot_x(x)

    def compute_q_(self, a: Array, b: Array) -> Tuple[float, float, float]:
        """
        Compute the 3 q terms for the simple quadratic constraint using dense matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float]
            q terms for the constraint

        Notes
        -----
        These expressions are the nonzero q terms defined in equation 2.45 in Pakman and Paninski (2014)
        """
        A = self.A
        c = self.c
        q1 = to_scalar(b.T @ A @ b - a.T @ A @ a)
        q3 = c + to_scalar(a.T @ A @ a)
        q4 = to_scalar(2 * a.T @ A @ b)
        return q1, q3, q4
    
    def compute_q_sparse(self, a: Array, b: Array) -> Tuple[float, float, float]:
        """
        Compute the 3 q terms for the simple quadratic constraint using sparse matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float]
            q terms for the constraint

        Notes
        -----
        These expressions are the nonzero q terms defined in equation 2.45 in Pakman and Paninski (2014)
        """
        q1 = to_scalar(self.x_dot_A_dot_x(b) - self.x_dot_A_dot_x(a))
        q3 = self.c + to_scalar(self.x_dot_A_dot_x(a))
        q4 = to_scalar(2 * a.T @ self.A_dot_x(b))
        return q1, q3, q4
    
    def hit_time(self, x: Array, xdot: Array) -> Array:
        """
        Compute the hit time for the simple quadratic constraint

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            The hit time for the constraint

        Notes
        -----
        Hit time is computed by solving Eqn 2.45 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.
        """
        a, b = xdot, x
        q1, q3, q4 = self.compute_q(a, b)
        u = np.sqrt(q1**2 + q4**2)
        if (u == 0) or (q4 == 0): 
            # No intersection so return NaN
            return np.array([np.nan])
        s1 = (np.pi + np.arcsin((q1+2*q3)/u) -
              np.arctan(q1/q4) + pis) / 2 
        s2 = (-np.arcsin((q1+2*q3)/u) -
              np.arctan(q1/q4)+ pis) / 2 
        s = np.hstack([s1, s2])
        return s[s > eps]



class QuadraticConstraint(BaseQuadraticConstraint):
    """
    Constraint of the form x**T A x + b**T x + c >= 0
    """
    def __init__(self, A: Array, b: Array, c: float, S: Array, sparse: bool = True, compiled: bool = True):
        """
        Parameters
        ----------
        A : Array
            The quadratic term matrix
        b : Array
            The linear term vector
        c : float
            The constant term
        S : Array
            The transformation matrix given by the Symmetric Sqrt of the Mass matrix
        sparse : bool
            Whether to use sparse matrix computations, by default True
        compiled : bool
            Whether to use compiled code, by default True

        Notes
        -----
        If A is a sparse matrix, sparse computations are used regardless of the
        sparse parameter.
        It is highly recommended to use compiled code for performance reasons.
        """
        self.c = c
        self.b = b
        if isinstance(A, Sparse):
            sparse = True
        self.sparse = sparse or compiled
        self.compiled = compiled

        if self.sparse:
            self._setup_values_sparse(A, S)
            self.S = S
        else:
            self._setup_values(A, S)
        if self.compiled:
            self.hit_time = self.hit_time_cpp
        else:
            self.hit_time = self.hit_time_py

    @classmethod 
    def build_from_dict(cls, d: dict, gpu: bool) -> 'QuadraticConstraint':
        """
        Build a QuadraticConstraint from a dictionary representation

        Parameters
        ----------
        d : dict
            Dictionary representation of the constraint
        gpu : bool
            Whether to load tensors onto the GPU

        Returns
        -------
        QuadraticConstraint
            The constructed constraint
        """
        if gpu and not _TORCH_AVAILABLE:
            gpu = False
            warnings.warn("GPU requested but PyTorch is not available. Loading on CPU instead.")
        sparse = d['sparse']
        A = d['A_orig']
        c = d['c']
        b = d['b']
        S = d.get('S', None)
        
        # Move to GPU if requested
        if gpu:
            if isinstance(S, Tensor):
                S = S.cuda()
            if isinstance(b, Tensor):
                b = b.cuda()
            if isinstance(A, Tensor):
                A = A.cuda()
        
        return cls(A, b, c, S, sparse, d.get('compiled', True))
    
    def value_(self, x: Array) -> float:
        """
        Compute the value of the constraint at x using dense matrix computations
        
        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        float
            The value of the constraint at x given by x^T A x + b^T x + c
        """
        return to_scalar(x.T @ self.A @ x + self.b.T @ x + self.c)
    
    def value_sparse(self, x: Array) -> float:
        """
        Compute the value of the constraint at x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            The value of the constraint at x given by x^T A x + b^T x + c
        """
        return to_scalar(self.x_dot_A_dot_x(x) + self.b.T @ x + self.c)

    def normal_(self, x: Array) -> Array:
        """
        Compute the normal vector at x using dense matrix computations
        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x + b
        """
        return 2 * self.A @ x + self.b
    
    def normal_sparse(self, x: Array) -> Array:
        """
        Compute the normal vector at x using sparse matrix computations
        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x + b
        """
        return 2 * self.A_dot_x(x) + self.b

    def compute_q_(self, a: Array, b: Array) -> Tuple[float, float, float, float, float]:
        """
        Compute the 5 q terms for the quadratic constraint using dense matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float, float, float]
            q terms for the quadratic constraint

        Notes
        -----
        These expressions are defined in Eqns 2.40-2.44 in Pakman and Paninski (2014)
        """
        A = self.A
        B = self.b
        c = self.c
        q1 = to_scalar(b.T @ A @ b - a.T @ A @ a)
        q2 = to_scalar(B.T @ b)
        q3 = c + to_scalar(a.T @ A @ a)
        q4 = to_scalar(2 * a.T @ A @ b)
        q5 = to_scalar(B.T @ a)
        return q1, q2, q3, q4, q5
    
    def compute_q_sparse(self, a: Array, b: Array) -> Tuple[float, float, float, float, float]:
        """
        Compute the 5 q terms for the quadratic constraint using sparse matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float, float, float]
            q terms for the quadratic constraint

        Notes
        -----
        These expressions are defined in Eqns 2.40-2.44 in Pakman and Paninski (2014)
        """
        B = self.b
        c = self.c
        q1 = to_scalar(self.x_dot_A_dot_x(b) - self.x_dot_A_dot_x(a))
        q2 = to_scalar(B.T @ b)
        q3 = c + to_scalar(self.x_dot_A_dot_x(a))
        q4 = to_scalar(2 * a.T @ self.A_dot_x(b))
        q5 = to_scalar(B.T @ a)
        return q1, q2, q3, q4, q5

    def hit_time_cpp(self, x: Array, xdot: Array) -> Array:
        """
        Compute the hit time for the quadratic constraint using compiled code

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            The hit time for the constraint

        Notes
        -----
        Hit time is computed by solving Eqn 2.48 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.

        Compiled code is both written in C++ and optimized to remove all redundant computations
        see paper for details.
        """
        a, b = xdot, x
        pis = np.array([-2, 0, 2]).reshape(-1,1)*np.pi
        qs = self.compute_q(a, b)
        # Old ctypes version --- IGNORE ---
        # soln = lib.calc_all_solutions(*qs)
        # s = np.ctypeslib.as_array(soln, shape=(1,8))
        # lib.free_ptr(soln)

        # New pybind11 compiled version
        s = calc_all_solutions(*qs).reshape((1,8))
        s = (s + pis).flatten()
        return np.unique(s[s > 1e-7])
    
    def hit_time_py(self, x: Array, xdot: Array) -> Array:
        """
        Compute the hit time for the quadratic constraint using Python code

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            The hit time for the constraint

        Notes
        -----
        Hit time is computed by solving Eqn 2.48 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.

        It is highly recommended to use the compiled version for performance reasons. 
        This Python version is maintained for testing and validation purposes.
        """
        a, b = xdot, x
        pis = np.array([-2, 0, 2]).reshape(-1,1)*np.pi
        qs = self.compute_q(a, b)
        s1 = soln1(*qs) + pis
        s2 = soln2(*qs) + pis
        s3 = soln3(*qs) + pis
        s4 = soln4(*qs) + pis
        s5 = soln5(*qs) + pis
        s6 = soln6(*qs) + pis
        s7 = soln7(*qs) + pis
        s8 = soln8(*qs) + pis
        s = np.hstack([s1, s2, s3, s4, s5, s6, s7, s8])
        return np.unique(s[s > 1e-7])
