from __future__ import annotations
import numpy as np
from typing import Tuple
from tmg_hmc.utils import Array, Sparse, to_scalar
from tmg_hmc.quad_solns import soln1, soln2, soln3, soln4, soln5, soln6, soln7, soln8
from tmg_hmc.gpu_utils import Tensor, _TORCH_AVAILABLE
from tmg_hmc.compiled import calc_all_solutions
from tmg_hmc.constraints.base import Constraint, BaseQuadraticConstraint, pis, eps
import warnings


@Constraint.register
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
            warnings.warn(
                "GPU requested but PyTorch is not available. Loading on CPU instead."
            )
        sparse = d["sparse"]
        A = d["A_orig"]
        c = d["c"]
        S = d.get("S", None)

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
        s1 = (np.pi + np.arcsin((q1 + 2 * q3) / u) - np.arctan(q1 / q4) + pis) / 2
        s2 = (-np.arcsin((q1 + 2 * q3) / u) - np.arctan(q1 / q4) + pis) / 2
        s = np.hstack([s1, s2])
        return s[s > eps]


@Constraint.register
class QuadraticConstraint(BaseQuadraticConstraint):
    """
    Constraint of the form x**T A x + b**T x + c >= 0
    """

    def __init__(
        self,
        A: Array,
        b: Array,
        c: float,
        S: Array,
        sparse: bool = True,
        compiled: bool = True,
    ):
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
    def build_from_dict(cls, d: dict, gpu: bool) -> "QuadraticConstraint":
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
            warnings.warn(
                "GPU requested but PyTorch is not available. Loading on CPU instead."
            )
        sparse = d["sparse"]
        A = d["A_orig"]
        c = d["c"]
        b = d["b"]
        S = d.get("S", None)

        # Move to GPU if requested
        if gpu:
            if isinstance(S, Tensor):
                S = S.cuda()
            if isinstance(b, Tensor):
                b = b.cuda()
            if isinstance(A, Tensor):
                A = A.cuda()

        return cls(A, b, c, S, sparse, d.get("compiled", True))

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

    def compute_q_(
        self, a: Array, b: Array
    ) -> Tuple[float, float, float, float, float]:
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

    def compute_q_sparse(
        self, a: Array, b: Array
    ) -> Tuple[float, float, float, float, float]:
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
        pis = np.array([-2, 0, 2]).reshape(-1, 1) * np.pi
        qs = self.compute_q(a, b)
        # Old ctypes version --- IGNORE ---
        # soln = lib.calc_all_solutions(*qs)
        # s = np.ctypeslib.as_array(soln, shape=(1,8))
        # lib.free_ptr(soln)

        # New pybind11 compiled version
        s = calc_all_solutions(*qs).reshape((1, 8))
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
        pis = np.array([-2, 0, 2]).reshape(-1, 1) * np.pi
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
