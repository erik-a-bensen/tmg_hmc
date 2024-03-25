import numpy as np
import scipy as sp
from typing.protocol import Protocol, Tuple

class Constraint(Protocol):
    def hit_time(self, a: np.ndarray, b: np.ndarray) -> float:
        pass

class LinearConstraint(Constraint):
    """
    Constraint of the form Ax + b >= 0
    """
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b

    def hit_time(self, a: np.ndarray, b: np.ndarray) -> float:
        pass 

class SimpleQuadraticConstraint(Constraint):
    """
    Constraint of the form x^T A x + c >= 0
    """
    def __init__(self, A: np.ndarray, c: float):
        self.A = A
        self.c = c

    def compute_q(self, a, b):
        A = self.A
        c = self.c
        q1 = b.T @ A @ b - a.T @ A @ b
        q3 = c + a.T @ A @ a
        q4 = 2 * a.T @ A @ b
        return q1, q3, q4

    def hit_time(self, a: np.ndarray, b: np.ndarray) -> float:
        q1, q3, q4 = self.compute_q(a, b)
        s1 = (np.pi - np.arcsin((-q1-2*q3)/(q1^2+q4^2)) -
              np.arctan2(q1, q4)) / 2
        s2 = (np.arcsin((-q1-2*q3)/(q1^2+q4^2)) -
              np.arctan2(q1, q4)) / 2
        s = np.array([s1, s2])
        return np.min(s[s > 0]), np.array() #TODO add final location

class QuadraticConstraint(Constraint):
    """
    Constraint of the form x^T A x + b^T x + c >= 0
    """
    def __init__(self, A: np.ndarray, b: np.ndarray, c: float):
        self.A = A
        self.b = b
        self.c = c

    def compute_q(self, a, b):
        A = self.A
        B = self.b
        c = self.c
        q1 = b.T @ A @ b - a.T @ A @ b
        q2 = B.T @ b
        q3 = c + a.T @ A @ a
        q4 = 2 * a.T @ A @ b
        q5 = B.T @ a
        return q1, q2, q3, q4, q5

    def hit_time(self, a: np.ndarray, b: np.ndarray) -> float:
        pass