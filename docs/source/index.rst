tmg-hmc
=======

**Exact Hamiltonian Monte Carlo sampling for truncated multivariate Gaussians with quadratic constraints**

This package implements the exact HMC algorithm from `Pakman and Paninski (2014)
<https://doi.org/10.1080/10618600.2013.788448>`_ for sampling from truncated
multivariate Gaussian distributions.

How It Works
------------

The algorithm uses Hamiltonian Monte Carlo with:

1. **Analytic Hamiltonian Dynamics**: Particles follow deterministic Hamiltonian
   trajectories that are analytically computable.
2. **Exact Bounces**: When a trajectory hits a constraint boundary, the algorithm
   computes the exact bounce time by solving the quartic equation for the hit time
   analytically.
3. **Perfect Acceptance Probability**: Unlike standard HMC, there is no integration
   error in solving the Hamiltonian dynamics, so the acceptance probability is always 1.

See `Pakman & Paninski (2014) <https://doi.org/10.1080/10618600.2013.788448>`_ for
mathematical details.

Features
--------

- **Flexible constraints** - Supports linear and quadratic inequality constraints
- **Efficient** - Uses optimized compiled C++ hit time calculation for efficient sampling
- **GPU acceleration** - Optional PyTorch backend for large-scale problems
- **Well-tested** - Comprehensive test suite ensuring correctness

Installation
------------

From PyPI:

.. code-block:: bash

   pip install tmg-hmc


With optional GPU support:

.. code-block:: bash

   pip install tmg-hmc[gpu]

From source:

.. code-block:: bash

   git clone https://github.com/erik-a-bensen/tmg_hmc.git
   cd tmg_hmc
   pip install .

**Requirements:** Python 3.10+, numpy, scipy. PyTorch is optional for GPU support.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   quickstart
   examples
   constraints
   api
   citing