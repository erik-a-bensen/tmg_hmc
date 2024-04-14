import numpy as np
from typing import Protocol, Tuple
#from tmg_hmc.utils_quartic import nanmin, soln1, soln2, soln3, soln4
from tmg_hmc.utils import nanmin, soln1, soln2, soln3, soln4

pis = np.array([-np.pi, 0, np.pi])
eps = 1e-8

class Constraint(Protocol):
    def value(self, x: np.ndarray) -> float:...
        
    def is_satisfied(self, x: np.ndarray) -> bool:
        return self.value(x) >= 0

    def is_zero(self, x: np.ndarray) -> bool:
        return np.isclose(self.value(x), 0)

    def hit_time(self, a: np.ndarray, b: np.ndarray) -> float:...

    def normal(self, x: np.ndarray) -> np.ndarray:...

    def reflect(self, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        f = self.normal(x)
        f = f / np.linalg.norm(f)
        return xdot - 2 * (f.T @ xdot) * f

class LinearConstraint(Constraint):
    """
    Constraint of the form fx + c >= 0
    """
    def __init__(self, f: np.ndarray, c: float):
        self.f = f
        self.c = c
    
    def value(self, x: np.ndarray) -> float:
        return self.f.T @ x + self.c

    def normal(self, x: np.ndarray) -> np.ndarray:
        return self.f

    def compute_q(self, a, b) -> Tuple[float, float]:
        f = self.f
        q1 = f.T @ a 
        q2 = f.T @ b
        return q1, q2

    def hit_time(self, x: np.ndarray, xdot: np.ndarray) -> float:
        a, b = xdot, x
        q1, q2 = self.compute_q(a, b)
        c = self.c
        u = np.sqrt(q1**2 + q2**2)
        if u < abs(c): # No intersection
            return np.nan
        s1 = -np.arccos(-c/u) + np.arctan(q1/q2) + pis
        s2 = np.arccos(-c/u) + np.arctan(q1/q2) + pis
        s = np.hstack([s1, s2])
        return nanmin(s[s > eps])

class SimpleQuadraticConstraint(Constraint):
    """
    Constraint of the form x**T A x + c >= 0
    """
    def __init__(self, A: np.ndarray, c: float):
        # Check that A is symmetric
        if not np.allclose(A, A.T):
            raise ValueError("A must be symmetric")
        self.A = A
        self.c = c
    
    def value(self, x: np.ndarray) -> float:
        return x.T @ self.A @ x + self.c

    def normal(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.A @ x

    def compute_q(self, a, b) -> Tuple[float, float, float]:
        A = self.A
        c = self.c
        q1 = b.T @ A @ b - a.T @ A @ a
        q3 = c + a.T @ A @ a
        q4 = 2 * a.T @ A @ b
        return q1, q3, q4

    def hit_time(self, x: np.ndarray, xdot: np.ndarray) -> float:
        a, b = xdot, x
        q1, q3, q4 = self.compute_q(a, b)
        u = np.sqrt(q1**2 + q4**2)
        s1 = (np.pi + np.arcsin((q1+2*q3)/u) -
              np.arctan(q1/q4) + pis) / 2 
        s2 = (-np.arcsin((q1+2*q3)/u) -
              np.arctan(q1/q4)+ pis) / 2 
        s = np.hstack([s1, s2])
        return nanmin(s[s > eps])

class QuadraticConstraint(Constraint):
    """
    Constraint of the form x**T A x + b**T x + c >= 0
    """
    def __init__(self, A: np.ndarray, b: np.ndarray, c: float):
        # Check that A is symmetric
        if not np.allclose(A, A.T):
            raise ValueError("A must be symmetric")
        self.A = A
        self.b = b
        self.c = c
    
    def value(self, x: np.ndarray) -> float:
        return x.T @ self.A @ x + self.b.T @ x + self.c

    def normal(self, x: np.ndarray) -> np.ndarray:
        return 2 * self.A @ x + self.b

    def compute_q(self, a, b) -> Tuple[float, float, float, float, float]:
        A = self.A
        B = self.b
        c = self.c
        q1 = b.T @ A @ b - a.T @ A @ a
        q2 = B.T @ b
        q3 = c + a.T @ A @ a
        q4 = 2 * a.T @ A @ b
        q5 = B.T @ a
        return q1, q2, q3, q4, q5

    def hit_time(self, x: np.ndarray, xdot: np.ndarray) -> float:
        a, b = xdot, x
        pis = np.array([-2*np.pi, 0, 2*np.pi])
        qs = self.compute_q(a, b)
        s1 = soln1(*qs) + pis#np.arccos(soln1(*qs)) + pis/2
        s2 = soln2(*qs) + pis#np.arccos(soln2(*qs)) + pis/2
        s3 = soln3(*qs) + pis#np.arccos(soln3(*qs)) + pis/2
        s4 = soln4(*qs) + pis#np.arccos(soln4(*qs)) + pis/2
        s = np.hstack([s1, s2, s3, s4])
        print(f"s: {s}")
        return nanmin(s[s > eps])
