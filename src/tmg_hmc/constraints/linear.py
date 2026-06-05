from __future__ import annotations
import numpy as np
from typing import Tuple
from tmg_hmc.utils import Array, to_scalar
from tmg_hmc.gpu_utils import _TORCH_AVAILABLE
from tmg_hmc.constraints.base import Constraint, pis, eps
import warnings


@Constraint.register
class LinearConstraint(Constraint):
    r"""
    Constraint of the form :math:`\mathbf{f}^T \mathbf{x} + c \geq 0`
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

    @classmethod
    def build_from_dict(cls, d: dict, gpu: bool) -> LinearConstraint:
        """
        Build a LinearConstraint from a dictionary representation

        Parameters
        ----------
        d : dict
            Dictionary representation of the constraint
        gpu : bool
            Whether to load tensors onto the GPU

        Returns
        -------
        Linear Constraint
            The constructed constraint
        """
        if gpu and not _TORCH_AVAILABLE:
            gpu = False
            warnings.warn(
                "GPU requested but PyTorch is not available. Loading on CPU instead."
            )
        f = d["f"]
        c = d["c"]

        return cls(f, c)

    def value(self, x: Array) -> float:
        r"""
        Compute the value of the constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the constraint at :math:`\mathbf{x}` given by
            :math:`\mathbf{f}^T \mathbf{x} + c`
        """
        return to_scalar(self.f.T @ x + self.c)

    def normal(self, x: Array) -> Array:
        r"""
        Compute the normal vector of the constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector of the constraint at :math:`\mathbf{x}` given by
            :math:`\mathbf{f}`
        """
        return self.f

    def compute_q(self, a: Array, b: Array) -> Tuple[float, float]:
        r"""
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
        simplifies to:

        .. math::

            q_1 \sin(t) + q_2 \cos(t) + c = 0
        """
        f = self.f
        q1 = to_scalar(f.T @ a)
        q2 = to_scalar(f.T @ b)
        return q1, q2

    def hit_time(self, x: Array, xdot: Array) -> Array:
        r"""
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
        the solution :math:`\pm \pi` to ensure we capture all hit times.

        Only positive hit times are returned and any ghost solutions are filtered
        out at a later stage.
        """
        q1, q2 = self.compute_q(xdot, x)
        c = self.c
        u = np.sqrt(q1**2 + q2**2)
        if (u < abs(c)) or (u == 0) or (q2 == 0):
            # No intersection so return NaN
            return np.array([np.nan])
        s1 = -np.arccos(-c / u) + np.arctan(q1 / q2) + pis
        s2 = np.arccos(-c / u) + np.arctan(q1 / q2) + pis
        s = np.hstack([s1, s2])
        return s[s > eps]
