import numpy as np
from scipy.sparse import csc_matrix
from typing import Protocol, Tuple

from tmg_hmc.constraints import Constraint, LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint
from tmg_hmc.utils_quartic import nanmin, nanargmin

class TMGSampler:
    """
    Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions
    with linear and quadratic constraints.
    """
    def __init__(self, mu: np.ndarray, Sigma: np.ndarray, T: float = np.pi/2) -> None:
        self.dim = len(mu)
        self.mu = mu.reshape(self.dim, 1)
        self.Sigma = Sigma
        self.s, self.V = np.linalg.eigh(Sigma)
        self.T = T
        self.constraints = []
        # Checks 
        if not np.shape(Sigma) == (self.dim, self.dim):
            raise ValueError("Sigma must be a square matrix")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Sigma must be symmetric")
        evals = np.linalg.eigvalsh(Sigma)
        if not np.all(evals >= 0 | np.isclose(evals, 0, atol=1e-8)):
            raise ValueError("Sigma must be positive semi-definite")
        
    def add_constraint(self, *, A: np.ndarray = None, f: np.ndarray = None, c: float = 0.0, sparse: bool = True) -> None:
        if f is not None:
            f = f.reshape(self.dim, 1)
            if sparse:
                f = csc_matrix(f)
        if A is not None:
            if not np.allclose(A, A.T):
                raise ValueError("A must be symmetric")
            if sparse:
                A = csc_matrix(A)
        
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
        times = np.array([c.hit_time(x, xdot) for c in self.constraints])
        ind = nanargmin(times)
        if ind is None:
            return np.nan, None
        return times[ind], self.constraints[ind]
    
    def _iterate(self, x: np.ndarray, xdot: np.ndarray, verbose: bool) -> np.ndarray:
        t = 0
        # print(f"Initial x: {x}")
        # print(f"Initial xdot: {xdot}")
        cprev = self.constraints[0]
        h, c = self._hit_time(x, xdot)
        # print(f"h: {h}")
        i = 0
        while not (h > self.T - t or np.isnan(h)):
            i += 1
            x, xdot = self._propagate(x, xdot, h)
            # print(f"x: {x}")
            # print(f"xdot: {xdot}")
            # if c.value(x) < -0.0:
            #     epsilon = 1e-10
            #     x, xdot = self._propagate(x, xdot, -epsilon)
            #     t -= epsilon
            # print(f"constraint value: {c.value(x)}")
            if c.is_zero(x):
                xdot = c.reflect(x, xdot)
            t += h
            cprev = c
            h, c = self._hit_time(x, xdot)
        #     print(f"h: {h}")
        # print(f"trem: {self.T - t}")
        # print(f"x: {x}")
        # print(f"xdot: {xdot}")
        x, xdot = self._propagate(x, xdot, self.T - t)
        # print(f"final x: {x}")
        if verbose:
            print(f"Number of collision checks: {i}")
        if not self._constraints_satisfied(x):
            # print(f"cprev: {cprev.value(x)}")
            raise ValueError("Error at final step")
        return x
    
    def sample_xdot(self) -> np.ndarray:
        return v * np.sqrt(s) * np.random.standard_normal(self.dim)
            
    def sample(self, x0: np.ndarray, n_samples: int, burn_in: int = 100, verbose=False) -> np.ndarray:
        x0 = x0.reshape(self.dim, 1)
        if not self._constraints_satisfied(x0):
            raise ValueError("Initial point does not satisfy constraints")
        samples = np.zeros((n_samples, self.dim))
        x = x0
        for i in range(burn_in):
            if verbose:
                print(f"burn-in iteration: {i+1} of {burn_in}")
            xdot = np.random.multivariate_normal(np.zeros(self.dim), self.Sigma).reshape(self.dim, 1)
            x = self._iterate(x, xdot, verbose)
        for i in range(n_samples):
            if verbose:
                print(f"sample iteration: {i+1} of {n_samples}")
            xdot = np.random.multivariate_normal(np.zeros(self.dim), self.Sigma).reshape(self.dim, 1)
            x = self._iterate(x, xdot, verbose)
            samples[i,:] = x.flatten()
        return samples