from __future__ import annotations
import numpy as np
from typing import Tuple, Protocol
from tmg_hmc.utils import Array, get_sparse_elements, to_scalar
from tmg_hmc.gpu_utils import torch, Tensor
from typing import runtime_checkable

pis = np.array([-1, 0, 1]) * np.pi
eps = 1e-12


@runtime_checkable
class Constraint(Protocol):
    """
    Abstract base class for constraints
    """

    _registry: dict = {}

    @classmethod
    def register(cls, subclass):
        Constraint._registry[subclass.__name__] = subclass
        return subclass  # return it so it can be used as a decorator

    def value(self, x: Array) -> float:
        """
        Compute the value of the constraint at x
        """
        ...

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
        return bool(np.isclose(val, 0)), bool(np.isclose(val, 0, atol=1e-2))

    def compute_q(self, a: Array, b: Array) -> Tuple[float, ...]:
        """
        Compute the coefficients of the constraint equation along the trajectory defined by a and b
        """
        ...

    def hit_time(self, a: Array, b: Array) -> Array:
        """
        Compute the hit time of the constraint along the trajectory defined by a and b
        """
        ...

    def normal(self, x: Array) -> Array:
        """
        Compute the normal vector of the constraint at x
        """
        ...

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
        if isinstance(f, Tensor):
            norm = torch.sqrt(f.T @ f)
            f = f / norm
        else:
            norm = np.sqrt(float(f.T @ f))
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
        if "sparse" in d and d["sparse"]:
            # Keep S if it exists
            if "S" not in d and hasattr(self, "S"):
                d["S"] = self.S
            # Remove problematic sparse reconstruction data
            keys_to_remove = ["s_rows", "s_cols", "row_data", "col_data"]
            for key in keys_to_remove:
                if key in d:
                    del d[key]

        # Convert tensors to CPU
        for k, v in d.items():
            if isinstance(v, Tensor):
                d[k] = v.cpu()

        d["type"] = self.__class__.__name__
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
        constraint_type = d["type"]
        if constraint_type not in cls._registry:
            raise ValueError(f"Unknown constraint type {constraint_type}")
        return cls._registry[constraint_type].build_from_dict(d, gpu)


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
        weighted = [
            normals[i] * float(np.prod(vals[:i] + vals[i + 1 :]))
            for i in range(len(self.constraints))
        ]
        if not weighted:
            raise ValueError("ProductConstraint has no constraints")
        return sum(weighted[1:], weighted[0])

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

    def compute_q(self, a: Array, b: Array) -> Tuple[float, ...]:
        raise NotImplementedError(
            "ProductConstraint does not support compute_q directly"
        )


class BaseQuadraticConstraint(Constraint):
    """
    Base class for quadratic constraints
    """

    def __init__(self) -> None:
        self.compute_type = "dense"

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
        self.compute_type = "dense"

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
        self.s_rows = [
            S[i, :].reshape((1, self.n)) for i in rows
        ]  # S[i,:] is a row vector
        self.s_cols = [
            S[:, j].reshape((self.n, 1)) for j in cols
        ]  # S[:,j] is a column vector
        self.a_vals = vals.reshape((self.n_comps,))
        self.compute_type = "sparse"

    def value_(self, x: Array) -> float:
        """Placeholder method for dense value computation"""
        raise NotImplementedError

    def value_sparse(self, x: Array) -> float:
        """Placeholder method for sparse value computation"""
        raise NotImplementedError

    def value(self, x: Array) -> float:
        """Dispatch method for value computation based on compute_type"""
        if self.compute_type == "dense":
            return self.value_(x)
        elif self.compute_type == "sparse":
            return self.value_sparse(x)
        else:
            raise ValueError(f"Unknown compute type {self.compute_type}")

    def normal_(self, x: Array) -> Array:
        """Placeholder method for dense normal vector computation"""
        raise NotImplementedError

    def normal_sparse(self, x: Array) -> Array:
        """Placeholder method for sparse normal vector computation"""
        raise NotImplementedError

    def normal(self, x: Array) -> Array:
        """Dispatch method for normal vector computation based on compute_type"""
        if self.compute_type == "dense":
            return self.normal_(x)
        elif self.compute_type == "sparse":
            return self.normal_sparse(x)
        else:
            raise ValueError(f"Unknown compute type {self.compute_type}")

    def compute_q_(self, a: Array, b: Array) -> Tuple[float, ...]:
        """Placeholder method for dense q term computation"""
        raise NotImplementedError

    def compute_q_sparse(self, a: Array, b: Array) -> Tuple[float, ...]:
        """Placeholder method for sparse q term computation"""
        raise NotImplementedError

    def compute_q(self, a: Array, b: Array) -> Tuple[float, ...]:
        """Dispatch method for q term computation based on compute_type"""
        if self.compute_type == "dense":
            return self.compute_q_(a, b)
        elif self.compute_type == "sparse":
            return self.compute_q_sparse(a, b)
        else:
            raise ValueError(f"Unknown compute type {self.compute_type}")

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
        dot_prods = [
            self.s_rows[i].reshape((1, self.n)) @ x for i in range(self.n_comps)
        ]
        terms = [
            self.a_vals[i] * dot_prods[i] * self.s_cols[i].reshape((self.n, 1))
            for i in range(self.n_comps)
        ]
        if not terms:
            raise ValueError("No components to sum")
        return sum(terms[1:], terms[0])

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
        return to_scalar(x.T @ self.A_dot_x(x))
