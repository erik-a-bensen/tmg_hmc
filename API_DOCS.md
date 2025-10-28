# TMG HMC API Documentation

Auto-generated from Python docstrings.

**Package:** `tmg_hmc` v0.0.2

**Description:** This package implements exact HMC sampling for truncated multivariate gaussians with quadratic constraints.

---

## Class: `TMGSampler`


    Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions
    with linear and quadratic constraints.
    

### `__init__(self, mu: 'Array' = None, Sigma: 'Array' = None, T: 'float' = 1.5707963267948966, gpu: 'bool' = False, *, Sigma_half: 'Array' = None) -> 'None'`


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
        

### `add_constraint(self, *, A: 'Array' = None, f: 'Array' = None, c: 'float' = 0.0, sparse: 'bool' = True, compiled: 'bool' = True) -> 'None'`


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
        

### `sample(self, x0: 'Array' = None, n_samples: 'int' = 100, burn_in: 'int' = 100, verbose=False, cont: 'bool' = False) -> 'Array'`


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
        

### `sample_xdot(self) -> 'Array'`


        Samples a new momentum vector xdot from the standard normal distribution handling GPU if necessary.
        

### `save(self, filename: 'str') -> 'None'`


        Saves the sampler state to a pickled file.
        

---

