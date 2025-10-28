# tmg_hmc

[![Python Versions](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Exact Hamiltonian Monte Carlo sampling for truncated multivariate Gaussians with quadratic constraints**

This package implements the exact HMC algorithm from [Pakman and Paninski (2014)](https://doi.org/10.1080/10618600.2013.788448) for sampling from truncated multivariate Gaussian distributions. 

## Features

- ✅ **Flexible constraints** - Supports linear and quadratic inequality constraints
- ✅ **Efficient** - Uses optimized compiled C++ hit time calculation for efficient sampling
- ✅ **GPU acceleration** - Optional PyTorch backend for large-scale problems
<!-- - ✅ **Well-tested** - Comprehensive test suite ensuring correctness -->

## Installation

### From Source
```bash
git clone https://github.com/erik-a-bensen/tmg_hmc.git
cd tmg_hmc 
pip install .
```

### Development Installation
```bash
git clone https://github.com/erik-a-bensen/tmg_hmc.git
cd tmg_hmc
pip install -e ".[dev]"  # Includes testing dependencies
```

**Requirements:**
- Python 3.10+
- numpy
- scipy
- torch

**Build Requirements:**
- C++ compiler (g++, clang, or MSVC)
- make (Unix-like systems)

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

print(f"Sample mean: {samples.mean(axis=1)}")
print(f"Sample covariance:\n{np.cov(samples)}")
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

## How It Works

The algorithm uses Hamiltonian Monte Carlo with
1. **Analytic Hamiltonian Dynamics**: Particles follow deterministic Hamiltonian trajectories that are analytically computable
2. **Exact Bounces**: When a trajectory hits a constraint boundary, the algorithm computes the exact bounce time by solving the quartic equation for the hit time analytically
4. **Perfect Acceptance Probability**: Unlike standard HMC, there's no integration error to solve the Hamiltonian dynamics. This means the acceptance probability is always 1.

See [Pakman & Paninski (2014)](https://doi.org/10.1080/10618600.2013.788448) for mathematical details.

## Performance Tips

- **GPU Acceleration**: For high-dimensional problems (d > 100), the PyTorch backend can provide significant speedups
- **Compiled Hit Times**: For quadratic constraints the optimized and compiled hit time calculation provides a significant speedup.

## Examples
TO BE IMPLEMENTED
<!-- See the `examples/` directory for:
- Linear constraint examples
- Quadratic constraint examples  
- High-dimensional sampling
- Comparison with approximate methods -->

## Testing

TO BE IMPLEMENTED
<!-- Run the test suite:
```bash
pytest tests/
``` -->

## Documentation

- [Full API Reference](API_DOCS.md) - Complete documentation of all functions and classes
- [Hit-time Calculations](resources/HMC_exact_soln.pdf) - Mathematica solutions for the hit times of each type of constraint.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

<!-- ## Related Projects
- [tmg](https://github.com/brunzema/truncated-mvn-sampler) - R package for approximate truncated Gaussian sampling -->

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

**Software:**
> Bensen, E. A. (2025). tmg_hmc: Exact HMC Sampling for Truncated Multivariate Gaussians with Quadratic Constraints. *TBD*. [To be updated upon acceptance]

**Methodology:**
> Pakman, A., & Paninski, L. (2014). Exact Hamiltonian Monte Carlo for Truncated Multivariate Gaussians. *Journal of Computational and Graphical Statistics*, 23(2), 518-542. https://doi.org/10.1080/10618600.2013.788448

<details>
<summary>BibTeX</summary>

```bibtex
@article{Bensen2025tmghmc,
  title={tmg\_hmc: Exact HMC Sampling for Truncated Multivariate Gaussians with Quadratic Constraints},
  author={Bensen, Erik A.},
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

## Acknowledgments

This implementation is based on the exact HMC algorithm developed by Ari Pakman and Liam Paninski.