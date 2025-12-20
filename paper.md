---
title: '`tmg-hmc`: A Python package for Exact HMC Sampling for Truncated Multivariate Gaussians with Linear and Quadratic Constraints'
tags:
  - Python
  - statistics
  - MCMC
  - HMC
  - truncated Multivariate Normal Distributions
authors:
  - name: Erik A. Bensen
    orcid: 0000-0002-2294-1421
    corresponding: true 
    affiliation: 1 
  - name: Mikael Kuusela
    orcid: 0000-0001-6875-945X
    affiliation: 1
affiliations:
 - name: Department of Statistics and Data Science, Carnegie Mellon University
   index: 1
date: 1 November 2025
bibliography: paper.bib
---

# Summary

Markov Chain Monte Carlo is a cornerstone of statistical methods that allows us to approximate intractable quantities by sampling from complex, multivariate probability distributions. One important class of distributions is constrained distributions. We present `tmg-hmc`: a Python implementation of the Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians with linear and quadratic inequality constraints introduced in @Pakman:2014. This method leverages the high-dimensional scalability and good mixing properties of Hamiltonian Monte Carlo while maintaining speed and simplicity since the Hamiltonian equations for a truncated Gaussian distribution are analytically solvable. This means that the sampler always accepts the sampled value and there are no tunable parameters. The original authors created an R implementation `tmg` and a Matlab implementation `hmc_tmg`. Both of these implementations are no longer maintained and the R package was archived from CRAN in 2021. There are also two R packages, `VeccTMVN` and `nntmvn`, that can sample truncated multivariate Gaussians. However these implementations are both approximate and limited to linear box constraints. To the best of our knowledge, `tmg-hmc` is the only existing Python implementation of Exact HMC. Additionally, we expand our implementation by including sparse matrix operations for sparse constraint handling and optional GPU acceleration for high-dimensional problems such as truncated Gaussian processes. Finally, we accelerate the quadratic constraint hit-time calculation by using a speed-optimized C++ implementation that can be called from Python.
 
# Statement of need

Many statistical models of real-world phenomena require the computation of intractable integrals over complex, multivariate probability distributions. Markov Chain Monte Carlo is a foundational statistical method that allows can be used to approximate these quantities from samples [@Robert:1999]. This has allowed for significant progress in statistical modeling in many areas of applied statistics and machine learning [@Gelman:2013]. One important class of distributions that arises due to parameter or data constraints are truncated distributions [@Gelfand:1992; @Swiler:2020; @Stanley:2025]. 
<!-- and more recently has helped enable the training of machine learning models for simulation based inference techniques [@Cranmer:2020; @Brehmer:2022] -->

@Pakman:2014 consider sampling a $d$-dimensional Gaussian $X \sim N(\mu,\Sigma)$ that is truncated with $m$ inequality constraints of the form 
$$Q_j(X) \geq 0,\quad\quad j=1,\hdots,m$$
where $Q_j(X)$ is a product of linear and quadratic polynomials. As discussed by Pakman and Paninski, this type of a distribution is a critical component of a vast array of statistical models including the probit and Tobit models [@Tobin:1958; @Albert:1993], the dichotomized Gaussian model [@Emrich:1991; @Cox:2002], stochastic integrate-and-fire neural models [@Paninski:2003], Bayesian isotonic regression [@Neelon:2004], and the Bayesian bridge model [@Polson:2014].

More recently distributions of this form have been used for learning partially censored Gaussian processes [@Cao:2025]. The Exact HMC algorithm was used to implement physics-informed constraints for fields governing CO2 flux in the WOMBAT v2.0 hierarchical flux-inversion framework for inferring changes to the global carbon cycle [@Bertolacci:2024]. Additionally, sampling constrained multivariate normals is relevant to a growing range of literature on constrained Gaussian processes [@Bachoc:2019; @Bachoc:2022; @Swiler:2020; @Agrell:2019; @DaVeiga:2012; @Lopez:2018; @Maatouk:2017]. In particular, in our ongoing research, we are using `tmg-hmc` to sample random transport maps given by the gradient of 2d convex Gaussian processes. We do this by approximating the convex GP by imposing quadratic convex inequality constraints on a discrete spatial grid. \autoref{fig:tmap} shows an example of such a transport map sampled using `tmg-hmc`.

![Sample of a random transport map defined as the gradient of a 2d convex Gaussian process.\label{fig:tmap}](./resources/example_tmap.png){ width=50% }

Exact HMC is not the only method for sampling distributions of this family. Two main alternatives include classical Hamiltonian Monte Carlo [@Duane:1987; @Neal:2011]and Gibbs sampling with the Hit-and-Run Algorithm [@Chen:1992]. HMC is a fast-mixing algorithm that is robust to high numbers of dimensions. However, generally speaking, it requires integrating equations of motion and using a Metropolis accept-reject step to account for numerical integration error. The numerical integration also comes with its own tunable hyperparameters that must be adjusted to balance exploration of the state space with a high acceptance probability [@Hoffman:2014]. On the other hand, Gibbs sampling is a simpler method with no hyperparameters that always accepts samples, however, it can be slow to mix, particularly when constraints impose high correlation between variables. Since the constrained Gaussian HMC trajectories are analytically computable Exact HMC enables the best of both options, the good mixing and high-dimensional capabilities of classical HMC with the always accepting and no hyperparameter properties of the Gibbs sampler. See the original manuscript @Pakman:2014 for a more detailed discussion of the differences between these methods. Some other alternative methods for sampling truncated multivariate Gaussian distributions include the R packages `VeccTMVN` [@Cao:2024] and `nntmvn` [@Cao:2025] which use Vecchia and nearest-neighbor approximations respectively to sample from a truncated Gaussian. However, these methods are both approximate and limited to sampling Gaussians with linear box constraints. `tmg-hmc` is developed as a flexible, user friendly and well tested Python package so that anyone can leverage the benefits of Exact HMC without needing to dwell on the technical details.

# Basic Usage

`tmg-hmc` operates predominantly through the `TMGSampler` class where a user specifies the untruncated distribution, adds constraints and then samples from the truncated distribution as illustrated in the example code below. All of the HMC trajectories and constraint hit-time solutions are handled automatically behind the scenes by the class internals.

```python
import numpy as np
from tmg_hmc import TMGSampler

# Set up untruncated distribution parameters
mu = np.array([0., 1.]).reshape(-1,1)
sigma = np.array([[1., 0.6],[0.6, 1.]])
sampler = TMGSampler(mu, sigma)

# Add constraints 
# Second coordinate positive
f_positivity = np.array([0., 1.]).reshape(-1,1)
c_positivity = 0
sampler.add_constraint(f=f_positivity, c=c_positivity)

# Bounded outside of unit circle
A_unit = np.eye(2)
c_unit = -1
sampler.add_constraint(A=A_unit, c=c_unit)

# Run the exact HMC sampling algorithm
x0 = np.array([2., 1.]).reshape(-1,1)
samples = sampler.sample(x0, n_samples=1000, burn_in=100)
```

# Acknowledgments

This work was partially supported by the U.S. National Science Foundation under Grant
No. DMS-2310632. 

# References