# tmg-hmc

![PyPI Version](https://img.shields.io/pypi/v/tmg_hmc)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/pypi/l/tmg_hmc)
[![Tests](https://github.com/erik-a-bensen/tmg_hmc/actions/workflows/run-tests.yml/badge.svg)](https://github.com/erik-a-bensen/tmg_hmc/actions/workflows/run-tests.yml)

> **Exact Hamiltonian Monte Carlo sampling for truncated multivariate Gaussians with quadratic constraints**

This package implements the exact HMC algorithm from [Pakman and Paninski (2014)](https://doi.org/10.1080/10618600.2013.788448) for sampling from truncated multivariate Gaussian distributions. 

## How It Works

The algorithm uses Hamiltonian Monte Carlo with
1. **Analytic Hamiltonian Dynamics**: Particles follow deterministic Hamiltonian trajectories that are analytically computable
2. **Exact Bounces**: When a trajectory hits a constraint boundary, the algorithm computes the exact bounce time by solving the quartic equation for the hit time analytically
4. **Perfect Acceptance Probability**: Unlike standard HMC, there's no integration error to solve the Hamiltonian dynamics. This means the acceptance probability is always 1.

See [Pakman & Paninski (2014)](https://doi.org/10.1080/10618600.2013.788448) for mathematical details.

## Features

- ✅ **Flexible constraints** - Supports linear and quadratic inequality constraints
- ✅ **Efficient** - Uses optimized compiled C++ hit time calculation for efficient sampling
- ✅ **GPU acceleration** - Optional PyTorch backend for large-scale problems
- ✅ **Well-tested** - Comprehensive test suite ensuring correctness

## Installation

### From PyPI
Base package install
```bash
pip install tmg-hmc
```

Optional GPU support
```bash
pip install tmg-hmc[gpu]
```

### From Source
Base package
```bash
git clone https://github.com/erik-a-bensen/tmg_hmc.git
cd tmg_hmc 
pip install .
```

Optional GPU support
```bash
git clone https://github.com/erik-a-bensen/tmg_hmc.git
cd tmg_hmc 
pip install .[gpu]
```

Optional Testing Dependencies
```bash
git clone https://github.com/erik-a-bensen/tmg_hmc.git
cd tmg_hmc 
pip install .[test] # Or .[all] for test + gpu
```

**Requirements:**
- Python 3.10+
- numpy
- scipy

**Optional GPU Requirements**
- torch

## Quick Start

### Linearly Constrained Gaussian
Sample a 2D standard normal with the y-component restricted to be positive:
```python
import numpy as np
from tmg_hmc import TMGSampler 

# Define the mean and covariance of the untruncated distribution
mu = np.zeros((2, 1))
Sigma = np.identity(2)
sampler = TMGSampler(mu, Sigma)

# Define the constraint y >= 0
# This corresponds to the constraint: f^T x + c >= 0
# where f = [0, 1] and c = 0
f = np.array([[0], [1]])
sampler.add_constraint(f=f, c=0)

# Sample 100 samples with 100 burn-in iterations
x0 = np.array([[1], [1]])  # Initial point (must satisfy constraints)
samples = sampler.sample(x0, n_samples=100, burn_in=100)
```

### Quadratically Constrained Gaussian
Sample from a Gaussian constrained to a circular region:
```python
import numpy as np
from tmg_hmc import TMGSampler

# 2D standard normal
mu = np.zeros((2, 1))
Sigma = np.identity(2)
sampler = TMGSampler(mu, Sigma)

# Add constraint: x^2 + y^2 <= 4 (inside circle of radius 2)
# Quadratic constraint: x^T A x + f^T x + c <= 0
# For x^2 + y^2 - 4 <= 0, we have A = I, f = 0, c = -4
A = np.identity(2)
c = -4
sampler.add_constraint(A=A, c=c)

# Sample
x0 = np.array([[0.5], [0.5]])
samples = sampler.sample(x0, n_samples=1000, burn_in=100)
```

### Multiple Constraints
Combine multiple constraints (e.g., box constraints):
```python
import numpy as np
from tmg_hmc import TMGSampler

mu = np.zeros((2, 1))
Sigma = np.identity(2)
sampler = TMGSampler(mu, Sigma)

# Box constraint: -1 <= x, y <= 1
# x >= -1  =>  [1,0]^T x + 1 >= 0
sampler.add_constraint(f=np.array([[1], [0]]), c=1)
# x <= 1   =>  [-1,0]^T x + 1 >= 0
sampler.add_constraint(f=np.array([[-1], [0]]), c=1)
# y >= -1  =>  [0,1]^T x + 1 >= 0
sampler.add_constraint(f=np.array([[0], [1]]), c=1)
# y <= 1   =>  [0,-1]^T x + 1 >= 0
sampler.add_constraint(f=np.array([[0], [-1]]), c=1)

x0 = np.array([[0], [0]])
samples = sampler.sample(x0, n_samples=1000, burn_in=100)
```

## Examples
See the `examples/` directory for:
- Linear constraint examples
- Quadratic constraint examples  
- Product constraint examples
- Truncated Gaussian process examples

## Testing

### Quick Start

Install test dependencies:
```bash
pip install -e .[test]
```

Run all tests:
```bash
pytest
```

### Running Specific Tests
```bash
pytest tests/test_sampler.py  # Run specific test file
pytest -v                     # Verbose output
pytest -m "not gpu"           # Skip GPU tests
pytest -m "gpu"               # GPU tests only
```

### Test Organization

- CPU tests run automatically in CI on Python 3.10, 3.11, 3.12, 3.13, 3.14
- GPU tests are automatically skipped if CUDA is not available
- Tests run on Ubuntu, Windows, and macOS

See the [Actions tab](https://github.com/erik-a-bensen/tmg_hmc/actions) for CI status.


## Documentation

- [Full API Reference](API_DOCS.md) - Complete documentation of all functions and classes
- [Hit-time Calculations](resources/HMC_exact_soln.pdf) - Mathematica solutions for the hit times of each type of constraint.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

**Software:**
> Bensen, E. A., & Kuusela, M. (2025). tmg_hmc: A Python package for Exact HMC Sampling for Truncated Multivariate Gaussians with Linear and Quadratic Constraints. *TBD*. [To be updated upon acceptance]

**Methodology:**
> Pakman, A., & Paninski, L. (2014). Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians. *Journal of Computational and Graphical Statistics*, 23(2), 518-542. https://doi.org/10.1080/10618600.2013.788448

<details>
<summary>BibTeX</summary>

```bibtex
@article{Bensen2025tmghmc,
  title={tmg\_hmc: A Python package for Exact HMC Sampling for Truncated Multivariate Gaussians with Linear and Quadratic Constraints},
  author={Bensen, Erik A. and Kuusela M.},
  journal={TBD},
  year={2025},
  note={[To be updated upon acceptance]}
}

@article{PakmanPaninski2014,
  title={Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians},
  author={Pakman, Ari and Paninski, Liam},
  journal={Journal of Computational and Graphical Statistics},
  volume={23},
  number={2},
  pages={518--542},
  year={2014},
  publisher={Taylor \& Francis},
  doi={10.1080/10618600.2013.788448}
}
```
</details>
