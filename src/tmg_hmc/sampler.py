from __future__ import annotations
import numpy as np
from typing import Tuple
from tmg_hmc.constraints import Constraint, LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint
from tmg_hmc.utils import Array, sparsify
import torch
import pickle

class TMGSampler:
    """
    Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions
    with linear and quadratic constraints.
    """
    def __init__(self, mu: Array = None, Sigma: Array = None, T: float = np.pi/2, gpu: bool = False,*,Sigma_half: Array = None) -> None:
        if Sigma is None and Sigma_half is None:
            raise ValueError("Must provide either Sigma or Sigma_half")
        self.dim = len(Sigma) if Sigma is not None else len(Sigma_half)
        if mu is None:
            mu = np.zeros(self.dim)
        self.mu = mu.reshape(self.dim, 1)
        self.T = T
        self.constraints = []
        self.constraint_violations = 0
        self.gpu = gpu
        self.x = None
        
        if Sigma_half is not None:
            self._setup_sigma_half(Sigma_half)
        else:
            self._setup_sigma(Sigma)
        if self.gpu:
            self.mu = torch.tensor(self.mu).cuda()
            
    def _setup_sigma(self, Sigma: Array) -> None:
        if not np.shape(Sigma) == (self.dim, self.dim):
            raise ValueError("Sigma must be a square matrix")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Sigma must be symmetric")
        # print(f"Checking positive semi-definiteness of Sigma...")
        if self.gpu:
            Sigma = torch.tensor(Sigma).cuda()
            s, V = torch.linalg.eigh(Sigma)
            all_positive = torch.all(s >= 0)
        else:
            s, V = np.linalg.eigh(Sigma)
            all_positive = np.all(s >= 0)
        if not all_positive:
            
            min_eig = torch.min(s) if self.gpu else np.min(s)
            if abs(min_eig) < 1e-10:
                s -= 2*min_eig
            else:
                raise ValueError("Sigma must be positive semi-definite")
        if self.gpu:
            self.Sigma_half = V @ torch.diag(torch.sqrt(s)) @ V.T
        else:
            self.Sigma_half = V @ np.diag(np.sqrt(s)) @ V.T
        
    def _setup_sigma_half(self, Sigma_half: Array) -> None:
        if not np.shape(Sigma_half) == (self.dim, self.dim):
            raise ValueError("Sigma_half must be a square matrix")
        if not np.allclose(Sigma_half, Sigma_half.T):
            raise ValueError("Sigma_half must be symmetric")
        self.Sigma_half = torch.tensor(Sigma_half).cuda() if self.gpu else Sigma_half
        
    def add_constraint(self, *, A: Array = None, f: Array = None, c: float = 0.0, sparse: bool = True) -> None:
        S = self.Sigma_half
        mu = self.mu
        if f is not None:
            f = f.reshape(self.dim, 1) 
        if A is not None:
            if not np.allclose(A, A.T):
                raise ValueError("A must be symmetric")
        
        if self.gpu:
            if A is not None:
                A = torch.tensor(A).cuda()
            if f is not None:
                f = torch.tensor(f).cuda()

        if (A is not None) and sparse:
            A = sparsify(A)
        if (f is not None) and sparse:
            f = sparsify(f)
        
        # A_new = S @ A @ S
        if (A is not None) and (f is not None):
            Amu = A @ mu
            f_new = S @ Amu * 2 + S @ f
            c_new = c + mu.T @ Amu + mu.T @ f
        elif (A is not None) and (f is None):
            Amu = A @ mu
            #f_new = 2*S @ A @ mu
            f_new = S @ Amu * 2
            #c_new = c + mu.T @ A @ mu
            c_new = mu.T @ Amu + c
        elif (A is None) and (f is not None):
            f_new = S @ f
            c_new = c + mu.T @ f
        else:
            raise ValueError("Must provide either A or f")

        nonzero_A = A is not None
        nonzero_f = torch.any(f_new != 0) if self.gpu else np.any(f_new != 0)
        if self.gpu:
            c_new = c_new.item()
        else:
            c_new = c_new[0,0]
        
        if nonzero_A and nonzero_f:
            self.constraints.append(QuadraticConstraint(A, f_new, c_new, S, sparse))
        elif nonzero_A and (not nonzero_f):
            self.constraints.append(SimpleQuadraticConstraint(A, c_new, S, sparse))
        elif (not nonzero_A) and nonzero_f:
            self.constraints.append(LinearConstraint(f_new, c_new))
            
    def _constraints_satisfied(self, x: Array) -> bool:
        if len(self.constraints) == 0:
            return True
        return all([c.is_satisfied(x) for c in self.constraints])
    
    def _propagate(self, x: Array, xdot: Array, t: float) -> Tuple[Array, Array]:
        xnew = xdot * np.sin(t) + x * np.cos(t)
        xdotnew = xdot * np.cos(t) - x * np.sin(t)
        return xnew, xdotnew
    
    def _hit_times(self, x: Array, xdot: Array) -> Tuple[Array, Array]:
        if len(self.constraints) == 0:
            return np.array([np.nan]), np.array([None])
        times = []
        cs = []
        for c in self.constraints:
            t = c.hit_time(x, xdot)
            times.append(t)
            cs += [c] * len(t)
        times = np.hstack(times)
        nanind = np.isnan(times)
        times = times[~nanind]
        cs = np.array(cs)[~nanind]
        if len(times) == 0:
            return np.array([np.nan]), np.array([None])
        inds = np.argsort(times)
        return times[inds], cs[inds]
    
    def _binary_search(self, x: Array, xdot: Array, b1: float, b2: float, c: QuadraticConstraint) -> Tuple[Array, Array, float, bool]:
        x1, _ = self._propagate(x, xdot, b1)
        hmid = (b1 + b2) / 2
        xmid, xdotmid = self._propagate(x, xdot, hmid)
        if np.isclose(c.value(xmid), 0):
            return xmid, xdotmid, hmid, True
        if np.sign(c.value(xmid)) != np.sign(c.value(x1)):
            return self._binary_search(x, xdot, b1, hmid, c)
        return self._binary_search(x, xdot, hmid, b2, c)
    
    def _refine_hit_time(self, x: Array, xdot: Array, c: QuadraticConstraint) -> Tuple[Array, Array, float, bool]:
        sign = np.sign(c.value(x))
        h = 1e-3 * sign
        x_temp, _ = self._propagate(x, xdot, h)
        if np.sign(c.value(x_temp)) == sign:
            return x, xdot, 0, False
        return self._binary_search(x, xdot, 0, h, c)
    
    def _iterate(self, x: Array, xdot: Array, verbose: bool) -> Array:
        t = 0 
        i = 0
        x_init, xdot_init = x, xdot
        hs, cs = self._hit_times(x, xdot)
        h, c = hs[0], cs[0]
        while h < self.T - t:
            i += 1
            inds = hs < self.T - t
            hs = hs[inds]
            cs = cs[inds]
            for pos in range(len(hs)):
                h, c = hs[pos], cs[pos]
                x_temp, xdot_temp = self._propagate(x, xdot, h)
                zero, refine = c.is_zero(x_temp)
                if refine and (not zero):
                    x_temp, xdot_temp, h_adj, zero = self._refine_hit_time(x_temp, xdot_temp, c)
                    h += h_adj
                if zero:
                    x, xdot = x_temp, xdot_temp
                    xdot = c.reflect(x, xdot)
                    t += h
                    break
            else:
                break
            hs, cs = self._hit_times(x, xdot)
            h, c = hs[0], cs[0]
        x, xdot = self._propagate(x, xdot, self.T - t)
        if verbose:
            print(f"\tNumber of collision checks: {i}")
        if self._constraints_satisfied(x):
            return x
        self.constraint_violations += 1
        if verbose:
            print(f"Constraint violated, redoing iteration")
        xdot = self.sample_xdot()
        return self._iterate(x_init, xdot, verbose)
    
    def sample_xdot(self) -> Array:
        if self.gpu:
            return torch.randn(self.dim, 1, dtype=torch.float64).cuda()
        else:
            return np.random.standard_normal(self.dim).reshape(self.dim, 1)
            
    def sample(self, x0: Array = None, n_samples: int = 100, burn_in: int = 100, verbose=False, cont: bool = False) -> Array:
        if (not cont) and (x0 is not None):
            x0 = x0.reshape(self.dim, 1)
            if self.gpu:
                x0 = torch.tensor(x0).cuda()
                x0 = torch.linalg.solve(self.Sigma_half, x0 - self.mu)
            else:
                x0 = np.linalg.solve(self.Sigma_half, x0 - self.mu)
            if not self._constraints_satisfied(x0):
                raise ValueError("Initial point does not satisfy constraints")
            x = x0
            self.constraint_violations = 0
            for i in range(burn_in):
                if verbose:
                    print(f"burn-in iteration: {i+1} of {burn_in}")
                xdot = self.sample_xdot()
                x = self._iterate(x, xdot, verbose)
            self.x = x
            if verbose:
                print(f"Constraint violations: {self.constraint_violations}")
        elif (not cont) and (x0 is None):
            raise ValueError("Must provide initial point if not continuing")

        samples = np.zeros((n_samples, self.dim))
        for i in range(n_samples):
            if verbose:
                print(f"sample iteration: {i+1} of {n_samples}")
            xdot = self.sample_xdot()
            self.x = self._iterate(self.x, xdot, verbose)
            correlated_x = (self.Sigma_half @ self.x).flatten() + self.mu.flatten()
            if self.gpu:
                correlated_x = correlated_x.cpu().numpy()
            samples[i,:] = correlated_x
        if verbose:
            print(f"Constraint violations: {self.constraint_violations}")
        return samples

    def save(self, filename: str) -> None:
        d = self.__dict__.copy()
        d['constraints'] = [c.serialize() for c in d['constraints']]
        if self.gpu:
            d['mu'] = d['mu'].cpu().numpy()
            d['Sigma_half'] = d['Sigma_half'].cpu().numpy()
            d['x'] = d['x'].cpu().numpy()
        with open(filename, 'wb') as f:
            pickle.dump(d, f)

    @classmethod
    def load(cls, filename: str) -> TMGSampler:
        with open(filename, 'rb') as f:
            d = pickle.load(f)
        d['constraints'] = [Constraint.deserialize(c, d['gpu']) for c in d['constraints']]
        sampler = cls(mu=d['mu'], Sigma_half=d['Sigma_half'], T=d['T'], gpu=d['gpu'])
        sampler.constraints = d['constraints']
        if d['x'] is not None:
            if d['gpu']:
                sampler.x = torch.tensor(d['x']).cuda()
            else:
                sampler.x = d['x']
        return sampler