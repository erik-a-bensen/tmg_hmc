import numpy as np
import ray
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
        self.T = T
        self.constraints = []

        # Checks 
        if not np.shape(Sigma) == (self.dim, self.dim):
            raise ValueError("Sigma must be a square matrix")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Sigma must be symmetric")
        print(f"Checking positive semi-definiteness of Sigma...")
        self.s, self.V = np.linalg.eigh(Sigma)
        if not np.all(self.s >= 0):
            eps = 1e-12 # Tolerance for positive semi-definiteness
            print("Sigma is not positive semi-definite, correcting...")
            Sigma = Sigma + eps * np.eye(self.dim)
            self.s, self.V = np.linalg.eigh(Sigma)
            if not np.all(self.s >= 0):
                raise ValueError("Sigma must be positive semi-definite")
        self.Sigma_half = self.V @ np.diag(np.sqrt(self.s)) @ self.V.T
        self.Sigma_inv_half = np.linalg.inv(self.Sigma_half)
        
    def add_constraint(self, *, A: np.ndarray = None, f: np.ndarray = None, c: float = 0.0, sparse: bool = True) -> None:
        S = self.Sigma_half
        mu = self.mu
        if f is not None:
            f = f.reshape(self.dim, 1) 
        else:
            f = np.zeros((self.dim, 1))
        if A is not None:
            if not np.allclose(A, A.T):
                raise ValueError("A must be symmetric")
        else:
            A = np.zeros((self.dim, self.dim))

        A_new = S @ A @ S
        f_new = 2*S @ A @ mu + S @ f
        c_new = c + mu.T @ A @ mu + f.T @ mu

        nonzero_A = np.any(A_new != 0)
        nonzero_f = np.any(f_new != 0)

        if sparse:
            A_new = csc_matrix(A_new)
        
        if nonzero_A and nonzero_f:
            self.constraints.append(QuadraticConstraint(A_new, f_new, c_new))
        elif nonzero_A and (not nonzero_f):
            self.constraints.append(SimpleQuadraticConstraint(A_new, c_new))
        elif (not nonzero_A) and nonzero_f:
            self.constraints.append(LinearConstraint(f_new, c_new))
        else:
            raise ValueError("Must provide either A or f")
            
    def _constraints_satisfied(self, x: np.ndarray) -> bool:
        if len(self.constraints) == 0:
            return True
        return all([c.is_satisfied(x) for c in self.constraints])
    
    def _propagate(self, x: np.ndarray, xdot: np.ndarray, t: float) -> Tuple[np.ndarray, np.ndarray]:
        xnew = xdot * np.sin(t) + x * np.cos(t)
        xdotnew = xdot * np.cos(t) - x * np.sin(t)
        return xnew, xdotnew
    
    def _hit_times(self, x: np.ndarray, xdot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        times = []
        cs = []
        for c in self.constraints:
            t = c.hit_time(x, xdot)
            if len(t) > 0:
                times.append(t)
                cs += [c] * len(t)
        times = np.hstack(times)
        nanind = np.isnan(times) & (times < self.T)
        times = times[~nanind]
        cs = np.array(cs)[~nanind]
        if len(times) == 0:
            return np.array([np.nan]), np.array([None])
        inds = np.argsort(times)
        return times[inds], cs[inds]
    
    def _iterate(self, x: np.ndarray, xdot: np.ndarray, verbose: bool) -> np.ndarray:
        t = 0
        cprev = None     
        i = 0
        h = -1
        while h < self.T - t:
            i += 1
            hs, cs = self._hit_times(x, xdot)
            inds = hs <= self.T - t
            hs = hs[inds]
            cs = cs[inds]
            for pos in range(len(hs)):
                h, c = hs[pos], cs[pos]
                x_temp, xdot_temp = self._propagate(x, xdot, h)
                if c.is_zero(x_temp):
                    x, xdot = x_temp, xdot_temp
                    xdot = c.reflect(x, xdot)
                    t += h
                    break
            else:
                if pos > 0:
                    x, xdot = x_temp, xdot_temp
                    t += h
                else:
                    break
        x, xdot = self._propagate(x, xdot, self.T - t)
        if verbose:
            print(f"\tNumber of collision checks: {i}")
        if not self._constraints_satisfied(x):
            raise ValueError("Error at final step")
        return x
    
    def sample_xdot(self) -> np.ndarray:
        return np.random.standard_normal(self.dim).reshape(self.dim, 1)
            
    def sample(self, x0: np.ndarray, n_samples: int, burn_in: int = 100, verbose=False) -> np.ndarray:
        x0 = x0.reshape(self.dim, 1)
        x0 = self.Sigma_inv_half @ (x0 - self.mu)
        if not self._constraints_satisfied(x0):
            raise ValueError("Initial point does not satisfy constraints")
        samples = np.zeros((n_samples, self.dim))
        x = x0
        for i in range(burn_in):
            if verbose:
                print(f"burn-in iteration: {i+1} of {burn_in}")
            xdot = self.sample_xdot()
            x = self._iterate(x, xdot, verbose)
        for i in range(n_samples):
            if verbose:
                print(f"sample iteration: {i+1} of {n_samples}")
            xdot = self.sample_xdot()
            x = self._iterate(x, xdot, verbose)
            samples[i,:] = (self.Sigma_half@x).flatten() + self.mu.flatten()
        return samples