from __future__ import annotations
import numpy as np
from typing import Tuple
from tmg_hmc.constraints import Constraint, LinearConstraint, SimpleQuadraticConstraint, QuadraticConstraint, ProductConstraint
from tmg_hmc.utils import Array, sparsify, is_nonzero_array
import warnings
import pickle
from tmg_hmc import get_torch, get_tensor_type

torch, Tensor = get_torch(), get_tensor_type()

class TMGSampler:
    """
    Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions
    with linear and quadratic constraints.
    """
    def __init__(self, mu: Array = None, Sigma: Array = None, T: float = np.pi/2, gpu: bool = False,*,Sigma_half: Array = None) -> None:
        """
        Parameters
        ----------
        mu : Array, optional
            Mean vector of the Gaussian distribution. If None, defaults to zero vector.
        Sigma : Array
            Covariance matrix of the Gaussian distribution. Must be positive semi-definite.
            Do not provide if Sigma_half is given.
        T : float, optional
            Integration time for the Hamiltonian dynamics. Default is pi/2.
        gpu : bool, optional
            Whether to use GPU acceleration with PyTorch. Default is False.
        Sigma_half : Array, optional
            Matrix such that Sigma_half @ Sigma_half.T = Sigma. 
            If provided, Sigma is not needed.
        """
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
        """
        Sets up the Sigma_half matrix from the covariance matrix Sigma.
        Ensures that Sigma is positive semi-definite.
        
        Parameters
        ----------
        Sigma : Array
            Covariance matrix of the untruncated Gaussian distribution.

        Raises
        ------
        ValueError
            If Sigma is not square, symmetric, or positive semi-definite.

        Notes
        -----
        If Sigma has very small negative eigenvalues (due to numerical errors),
        they are shifted to ensure positive semi-definiteness.

        The symmetric square root of Sigma is computed using eigenvalue decomposition.
        This method was chosen because it is more numerically stable than Cholesky decomposition
        for positive semi-definite matrices that may be close to singular.
        """
        if not np.shape(Sigma) == (self.dim, self.dim):
            raise ValueError("Sigma must be a square matrix")
        if not np.allclose(Sigma, Sigma.T):
            raise ValueError("Sigma must be symmetric")
        
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
        """
        Sets up the Sigma_half matrix directly.

        Parameters
        ----------
        Sigma_half : Array
            Matrix such that Sigma_half @ Sigma_half.T = Sigma.

        Raises
        ------
        ValueError
            If Sigma_half is not square or symmetric.
        """
        if not np.shape(Sigma_half) == (self.dim, self.dim):
            raise ValueError("Sigma_half must be a square matrix")
        if not np.allclose(Sigma_half, Sigma_half.T):
            raise ValueError("Sigma_half must be symmetric")
        if self.gpu:
            Sigma_half = torch.tensor(Sigma_half).cuda()
            s, V = torch.linalg.eigh(Sigma_half)
            all_positive = torch.all(s >= 0)
        else:
            s, V = np.linalg.eigh(Sigma_half)
            all_positive = np.all(s >= 0)
        if not all_positive:
            min_eig = torch.min(s) if self.gpu else np.min(s)
            if abs(min_eig) < 1e-10:
                s -= 2*min_eig
            else:
                raise ValueError("Sigma_half must be positive semi-definite")
        if self.gpu:
            self.Sigma_half = V @ torch.diag(s) @ V.T
        else:
            self.Sigma_half = V @ np.diag(s) @ V.T

    def _build_constraint(self, *, A: Array = None, f: Array = None, c: float = 0.0, sparse: bool = True, compiled: bool = True) -> Constraint:
        """
        Builds a constraint to the sampler of the form:
            x.T @ A @ x + f.T @ x + c >= 0

        Parameters
        ----------
        A : Array, optional
            Quadratic term matrix, defaults to the zero matrix if not provided.
        f : Array, optional
            Linear term vector, defaults to the zero vector if not provided.
        c : float, optional
            Constant term. Default is 0.0.
        sparse : bool, optional
            Whether to store A and f in sparse format. Default is True.
        compiled : bool, optional
            Whether to use compiled constraint solutions for full quadratic constraints. Default is True.

        Raises
        ------
        ValueError
            If A is not symmetric when provided, or if neither A nor f is provided.

        Notes
        -----
        The constraint is automatically transformed to account for the Gaussian's mean and covariance.
        The transformed constraint becomes:
            y.T @ (S @ A @ S) @ y + (2 * S @ A @ mu + S @ f).T @ y + (mu.T @ A @ mu + mu.T @ f + c) >= 0
        where y = S^{-1} (x - mu) and S = Sigma_half.
        Depending on whether A and f are non-zero, the appropriate constraint type is chosen.
        """
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

        nonzero_A = False
        if A is not None:
            nonzero_A = is_nonzero_array(A)
        nonzero_f = is_nonzero_array(f_new)
        if self.gpu:
            c_new = c_new.item()
        else:
            c_new = c_new[0,0]
        
        if nonzero_A and nonzero_f:
            return QuadraticConstraint(A, f_new, c_new, S, sparse, compiled)
        elif nonzero_A and (not nonzero_f):
            return SimpleQuadraticConstraint(A, c_new, S, sparse)
        elif (not nonzero_A) and nonzero_f:
            return LinearConstraint(f_new, c_new)
        else:
            raise ValueError("Constraint cannot be trivial (A and f both zero after transformation)")
        
    def add_constraint(self, *, A: Array = None, f: Array = None, c: float = 0.0, sparse: bool = True, compiled: bool = True) -> None:
        """
        Adds a constraint to the sampler of the form:
            x.T @ A @ x + f.T @ x + c >= 0

        Parameters
        ----------
        A : Array, optional
            Quadratic term matrix, defaults to the zero matrix if not provided.
        f : Array, optional
            Linear term vector, defaults to the zero vector if not provided.
        c : float, optional
            Constant term. Default is 0.0.
        sparse : bool, optional
            Whether to store A and f in sparse format. Default is True.
        compiled : bool, optional
            Whether to use compiled constraint solutions for full quadratic constraints. Default is True.

        Raises
        ------
        ValueError
            If A is not symmetric when provided, or if neither A nor f is provided.

        Notes
        -----
        The constraint is automatically transformed to account for the Gaussian's mean and covariance.
        The transformed constraint becomes:
            y.T @ (S @ A @ S) @ y + (2 * S @ A @ mu + S @ f).T @ y + (mu.T @ A @ mu + mu.T @ f + c) >= 0
        where y = S^{-1} (x - mu) and S = Sigma_half.
        Depending on whether A and f are non-zero, the appropriate constraint type is chosen.
        """
        constraint = self._build_constraint(A=A, f=f, c=c, sparse=sparse, compiled=compiled)
        self.constraints.append(constraint)

    def add_product_constraint(self, *, parameters: list[list[Array]] | list[dict[str,Array]], sparse: bool = True, compiled: bool = True) -> None:
        """
        Adds a constraint to the sampler of the form:
            x.T @ A @ x + f.T @ x + c >= 0

        Parameters
        ----------
        parameters: list[list[Array]] | list[dict[str,Array]]
            List of constraint parameters as either lists [A, f, c] or dictionaries {'A': A, 'f': f, 'c': c}.
            If list, each element must be of length 3 corresponding to A, f, and c.
            If dictionary, missing keys 'A', 'f', and 'c' default to None, None, and 0.0 respectively.
        sparse : bool, optional
            Whether to store A and f in sparse format. Default is True.
        compiled : bool, optional
            Whether to use compiled constraint solutions for full quadratic constraints. Default is True.

        Raises
        ------
        ValueError
            If A is not symmetric when provided, or if neither A nor f is provided.

        Notes
        -----
        For product constraints, you must provide lists of each component (A, f, c).
        The constraint is automatically transformed to account for the Gaussian's mean and covariance.
        The transformed constraint becomes:
            y.T @ (S @ A @ S) @ y + (2 * S @ A @ mu + S @ f).T @ y + (mu.T @ A @ mu + mu.T @ f + c) >= 0
        where y = S^{-1} (x - mu) and S = Sigma_half.
        Depending on whether A and f are non-zero, the appropriate constraint type is chosen.
        """
        def parse_param(p):
            if isinstance(p, dict):
                A = p.get('A', None)
                f = p.get('f', None)
                c = p.get('c', 0.0)
            else:
                if len(p) != 3:
                    raise ValueError("Each parameter list must be of length 3 corresponding to A, f, and c")
                A, f, c = p
            return A, f, c
        if len(parameters) == 0:
            raise ValueError("Must provide at least one constraint parameter set")
        elif len(parameters) == 1:
            warnings.warn("Only one constraint provided, adding as regular constraint instead of product constraint", UserWarning)
            A, f, c = parse_param(parameters[0])
            constraint = self._build_constraint(A=A, f=f, c=c, sparse=sparse, compiled=compiled)
            self.constraints.append(constraint)
            return
        cs = []
        for p in parameters:
            A, f, c = parse_param(p)
            constraint = self._build_constraint(A=A, f=f, c=c, sparse=sparse, compiled=compiled)
            cs.append(constraint)
        product_constraint = ProductConstraint(cs)
        self.constraints.append(product_constraint)
            
    def _constraints_satisfied(self, x: Array) -> bool:
        """
        Checks if all constraints are satisfied at point x.
        
        Parameters
        ----------
        x : Array
            Point in the transformed space to check.

        Returns
        -------
        bool
            True if all constraints are satisfied, False otherwise.
        """
        if len(self.constraints) == 0:
            return True
        return all([c.is_satisfied(x) for c in self.constraints])
    
    def _propagate(self, x: Array, xdot: Array, t: float) -> Tuple[Array, Array]:
        """
        Propagates the state (x, xdot) forward in time by t according to the Hamiltonian dynamics
        of a standard Gaussian distribution.

        Parameters
        ----------
        x : Array
            Current position in the transformed space.
        xdot : Array
            Current momentum in the transformed space.
        t : float
            Time to propagate.

        Returns
        -------
        Tuple[Array, Array]
            New position and momentum after propagation.
        """
        xnew = xdot * np.sin(t) + x * np.cos(t)
        xdotnew = xdot * np.cos(t) - x * np.sin(t)
        return xnew, xdotnew
    
    def _hit_times(self, x: Array, xdot: Array) -> Tuple[Array, Array]:
        """
        Computes the hit times for all constraints given the current state (x, xdot).
        Returns sorted hit times and corresponding constraints.

        Parameters
        ----------
        x : Array
            Current position in the transformed space.
        xdot : Array
            Current momentum in the transformed space.

        Returns
        -------
        Tuple[Array, Array]
            Sorted hit times and corresponding constraints.
        """
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
    
    def _binary_search(self, x: Array, xdot: Array, b1: float, b2: float, c: Constraint) -> Tuple[Array, Array, float, bool]:
        """
        Performs a binary search to find the precise hit time for a constraint
        between bounds b1 and b2.
        
        Parameters
        ----------
        x : Array
            Current position in the transformed space.
        xdot : Array
            Current momentum in the transformed space.
        b1 : float
            Lower bound of the search interval.
        b2 : float
            Upper bound of the search interval.
        c : Constraint
            The constraint to check.

        Returns
        -------
        Tuple[Array, Array, float, bool]
            New position, new momentum, hit time, and a boolean indicating if the search was successful.
        """
        x1, _ = self._propagate(x, xdot, b1)
        hmid = (b1 + b2) / 2
        xmid, xdotmid = self._propagate(x, xdot, hmid)
        x2, _ = self._propagate(x, xdot, b2)
        if np.isclose(c.value(xmid),0., atol=1e-12):
            return xmid, xdotmid, hmid, True
        if np.sign(c.value(xmid)) != np.sign(c.value(x1)):
            return self._binary_search(x, xdot, b1, hmid, c)
        return self._binary_search(x, xdot, hmid, b2, c)
    
    def _refine_hit_time(self, x: Array, xdot: Array, c: QuadraticConstraint) -> Tuple[Array, Array, float, bool]:
        """
        Refines the hit time for a quadratic constraint by moving the position towards the constraint 
        boundary and performing a binary search.

        Parameters
        ----------
        x : Array
            Current position in the transformed space.
        xdot : Array
            Current momentum in the transformed space.
        c : QuadraticConstraint
            The quadratic constraint to refine.
        
        Returns
        -------
        Tuple[Array, Array, float, bool]
            New position, new momentum, adjusted hit time, and a boolean indicating if a hit was found.

        Notes
        -----
        Failure to find a hit indicates a ghost hit which is handled in _iterate.
        """
        value = c.value(x)
        sign = np.sign(value)
        h = 1e-3 * sign
        x_temp, _ = self._propagate(x, xdot, h)
        if np.sign(c.value(x_temp)) == sign:
            # If the refined position is still on the same side of the constraint, no hit was found
            return x, xdot, 0, False
        return self._binary_search(x, xdot, 0, h, c)
    
    def _iterate(self, x: Array, xdot: Array, verbose: bool = False) -> Array:
        """
        Performs a single iteration of the HMC sampler, propagating the state (x, xdot)
        and handling constraint collisions.

        Parameters
        ----------
        x : Array
            Current position in the transformed space.
        xdot : Array
            Current momentum in the transformed space.
        verbose : bool, optional
            Whether to print verbose output.

        Returns
        -------
        Array
            New position after the iteration.

        Notes
        -----
        This method handles refines hit times to improve accuracy and manages ghost hits.
        As a fallback, if constraints are violated after propagation, the iteration is 
        redone with a new momentum. However this is extremely rare.
        """
        t = 0 
        i = 0
        x_init = x
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
                    # Found ghost hit, continue to next hit time
                    continue
            else:
                # No hit times found before max integration time, so break out of while loop
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
        """
        Samples a new momentum vector xdot from the standard normal distribution handling GPU if necessary.
        """
        if self.gpu:
            return torch.randn(self.dim, 1, dtype=torch.float64).cuda()
        else:
            return np.random.standard_normal(self.dim).reshape(self.dim, 1)
            
    def sample(self, x0: Array = None, n_samples: int = 100, burn_in: int = 100, verbose=False, cont: bool = False) -> Array:
        """
        Generates samples from the truncated multivariate Gaussian distribution.

        Parameters
        ----------
        x0 : Array
            Initial point for the sampler. Optional if cont is True.
        n_samples : int, optional
            Number of samples to generate, default is 100.
        burn_in : int, optional
            Number of burn-in iterations, default is 100.
        verbose : bool, optional
            Whether to print verbose output, default is False.
        cont : bool, optional
            Whether to continue from the last sampled point. Default is False.

        Returns
        -------
        Array
            Generated samples.

        Raises
        ------
        ValueError
            If cont is False and x0 is not provided, or if x0 does not satisfy constraints.
        """
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
        """
        Saves the sampler state to a pickled file.
        """
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
        """
        Loads the sampler state from a pickled file.
        """
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