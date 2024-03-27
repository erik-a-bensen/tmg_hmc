import numpy as np
from typing import Protocol, Tuple

from tmv_hmc.constraints import Constraint, LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint

class TMVSampler:
    """
    Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions
    with linear and quadratic constraints.
    """
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, T: float = np.pi/2) -> None:
        self.dim = len(mu)
        self.mu = mu.reshape(self.dim, 1)
        self.Sigma = Sigma
        self.T = T
        self.constraints = []
        # Checks 
        if not np.shape(Sigma) == (self.dim, self.dim):
            raise ValueError("Sigma must be a square matrix")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Sigma must be symmetric")
        if not np.all(np.linalg.eigvals(Sigma) >= 0):
            raise ValueError("Sigma must be positive semi-definite")
        
    def add_constraint(self, *, A: np.ndarray = None, f: np.ndarray = None, c: float = 0.0) -> None:
        match [A, f]:
            case [np.ndarray, np.ndarray]:
                self.constraints.append(QuadraticConstraint(A, f, c))
            case [np.ndarray, None]:
                self.constraints.append(SimpleQuadraticConstraint(A, c))
            case [None, np.ndarray]:
                self.constraints.append(LinearConstraint(f, c))
            case [None, None]:
                raise ValueError("Must provide either A or f")
            
    def _constraints_satisfied(self, x: np.ndarray) -> bool:
        return all([c.is_satisfied(x) for c in self.constraints])
            
    def sample(self, n_samples: int, x0: np.ndarray, burn_in: int = 100) -> np.ndarray:
        if not self._constraints_satisfied(x0):
            raise ValueError("Initial point does not satisfy constraints")
        samples = np.zeros((n_samples, self.dim))