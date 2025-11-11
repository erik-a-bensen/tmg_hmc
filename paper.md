---
title: 'tmg_hmc: A Python package for Exact HMC Sampling for Truncated Multivariate Gaussians with Linear and Quadratic Constraints'
tags:
  - Python
  - statistics
  - MCMC
  - HMC
  - truncated distributions
authors:
  - name: Erik A. Bensen
    orcid: 0000-0002-2294-1421
    # equal-contrib: true
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Mikael Kuusela
    orcid: 0000-0001-6875-945X
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
#   - name: Author with no affiliation
#     affiliation: 3
#   - given-names: Ludwig
#     dropping-particle: van
#     surname: Beethoven
#     affiliation: 3
affiliations:
 - name: Carnegie Mellon University, United States
   index: 1
#  - name: Institution Name, Country
#    index: 2
#  - name: Independent Researcher, Country
#    index: 3
date: 1 November 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Markov Chain Monte Carlo is a cornerstone of statistical methods that allows us to approximate intractable quantities by sampling from complex multivariate probability distributions. One large class of important distributions are constrained distributions. We present `tmg_hmc`: a Python implementation of the Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians presented by Pakman and Paninski [@Pakman:2014]. This method leverages the high dimensional scalability and well mixing properties of Hamiltonian Monte Carlo while maintaining speed and simplicity since the Hamiltonian equations for a Gaussian distribution are analytically solvable. This means that the sampler always accepts and there are no tunable parameters. Our implementation replaces the existing R implementation `tmg` and matlab implementation `tmg_hmc` both of which are deprecated and no longer maintained. The R package was archived from CRAN in 2021 for this reason. Additionally, we expand our implementation by including sparse matrix operations for sparse constraint handling and optional GPU acceleration for high dimensional problems such as truncated Gaussian processes. Finally, we accelerate the Quadratic constraint hit time calculation by using a speed optimized C++ implementation and binding it to be called with Python.
 
# Statement of need

Markov Chain Monte Carlo has been a foundational statistical technique that allows the approximation of intractable quantities from samples of complex, multivarate probability distriutions [@Robert:1999]. This has allowed for significant progress in statistical modeling in many areas of applied statistics and machine learning [@Gelman:1995] and more recently has helped enable the training of machine learning models for simulation based inference techniques [@Cranmer:2020; Brehmer:2022]. One important class of distributions that arises due to parameter or data constraints are truncated distributions [@Gelfand:1992; @Stanley:2025]. 

Pakman and Paninski consider sampling a $d$-dimensional Gaussian $X \sim N(\mu,\Sigma)$ that is truncated with $m$ inequality constraints of the form 
$$Q_j(X) \geq 0\quad\quad j=1,\hdots,m$$
where $Q_j(X)$ is a product of linear and quadratic polynomials. As discussed by Pakman and Paninski, this type of distribution is critical to a vast array of Bayesian models including the probit and Tobit models [@Tobin:1958; @Albert:1993], the dichotomized Gaussian model [@Emrich:1991; Cox:2002], stochastic integrate-and-fire neural models [@Paninski:2003], Bayesian isotonic regression [@Neelon:2004], and the Bayesian bridge model [@Polson:2014].

Exact HMC is not the only method for sampling distributions of this family, two main alternatives include Hamiltonian Monte Carlo and Gibbs Sampling with the Hit and Run Algorithm [@Deely:1992]. HMC is a fast mixing algorithm that is robust to high numbers of dimensions. However, generally speaking it requires integrating equations of motion and using a Metropolis accept-reject step to account for numerical integration error. The numerical integration also comes with its own tunable hyperparameters that must be adjusted to balance exploration of the state space with a high acceptance probability [Hoffman:2024]. On the other hand, Gibbs is a simpler method with no hyperparameters that always accepts samples, however, it can be slow to mix, particularly when constraints impose high correlation between parameters. Since the Gaussian HMC trajectories are analytically computable Exact HMC enables the best of both options, the good mxing and high dimensional capabilities of HMC with the always accepting and no hyperparameter properties of the Gibbs sampler. See the original manuscript by Pakman and Paninski [@Pakman:2014] for a more detailed discussion of the differences between these methods. `tmg_hmc` is developed as a user friendly, well tested Python package so that anyone can leverage the benefits of exact HMC without needing to dwell on the technical details.

In recent years, Exact HMC was used to implement physics-informed constraints such as positivity/ negativity constraints for fields governing CO2 flux in the WOMBAT v2.0 hierarchical flux-inversion framework for inferring changes to the global carbon cycle [@Bertolacci:2024]. And this package, tmg_hmc is used in ongoing research to sample spatial warping errors derived from 2d convex Gaussian processes which are approximated using a sequence of quadratic inequality constraints [Cite our stuff].

<!-- 
# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% } -->

# Acknowledgements
<!-- 
We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project. -->
This implementation is based on the methodology presented by Pakman and Paninski [@Pakman:2014].

# References