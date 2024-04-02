import numpy as np
from typing import Protocol, Tuple

from tmg_hmc.constraints import Constraint, LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint
from tmg_hmc.utils import nanmin, nanargmin

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
        # Reshape f to be a column vector
        if f is not None:
            f = f.reshape(self.dim, 1)
        if A is not None and f is not None:
            self.constraints.append(QuadraticConstraint(A, f, c))
        elif A is not None and f is None:
            self.constraints.append(SimpleQuadraticConstraint(A, c))
        elif A is None and f is not None:
            self.constraints.append(LinearConstraint(f, c))
        else:
            raise ValueError("Must provide either A or f")
            
    def _constraints_satisfied(self, x: np.ndarray) -> bool:
        return all([c.is_satisfied(x) for c in self.constraints])
    
    def _propagate(self, x: np.ndarray, xdot: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        xnew = self.mu + xdot * np.sin(t) + (x - self.mu) * np.cos(t)
        xdotnew = xdot * np.cos(t) - (x - self.mu) * np.sin(t)
        return xnew, xdotnew
    
    def _hit_time(self, x: np.ndarray, xdot: np.ndarray) -> Tuple[float, Constraint]:
        times = [c.hit_time(x, xdot) for c in self.constraints]
        ind = nanargmin(times)
        if ind is None:
            return np.nan, None
        return times[ind], self.constraints[ind]
    
    def _iterate(self, x: np.ndarray, xdot: np.ndarray) -> np.ndarray:
        t = 0
        i = 0
        while True:
            i += 1
            h, c = self._hit_time(x, xdot)
            if h > self.T - t or np.isnan(h):
                break
            xdotprev = xdot
            x, xdot = self._propagate(x, xdot, h)
            xdot = c.reflect(x, xdot)
            t += h
            if c.value(x) < 0:
                epsilon = 1e-6
                print(f"constraint value: {c.value(x)}")
                print(f"x: {x.flatten()}")
                print(f"xdot reflected: {xdot.flatten()}")
                x, xdot = self._propagate(x, xdotprev, -epsilon)
                print(f"x hack corrected: {x.flatten()}")
                print(f"Corrected constraint value: {c.value(x)}")
                t -= epsilon
        x, xdot = self._propagate(x, xdot, self.T - t)
        return x
            
    def sample(self, x0: np.ndarray, n_samples: int, burn_in: int = 100) -> np.ndarray:
        if not self._constraints_satisfied(x0):
            raise ValueError("Initial point does not satisfy constraints")
        samples = np.zeros((n_samples, self.dim))
        x = x0
        for i in range(burn_in):
            xdot = np.random.multivariate_normal(np.zeros(self.dim), self.Sigma).reshape(self.dim, 1)
            x = self._iterate(x, xdot)
        for i in range(n_samples):
            xdot = np.random.multivariate_normal(np.zeros(self.dim), self.Sigma).reshape(self.dim, 1)
            x = self._iterate(x, xdot)
            samples[i,:] = x.flatten()
        return samples