# mpitools

<!-- ![PyPI Version](https://img.shields.io/pypi/v/mpi4pytools)
![Development Status](https://img.shields.io/pypi/status/mpi4pytools)
![Python Versions](https://img.shields.io/pypi/pyversions/mpi4pytools)
![License](https://img.shields.io/pypi/l/mpi4pytools)

> **⚠️ Development Notice**: This package is in active development. The API may change significantly between versions until v1.0.0. Use in production environments is not recommended. -->

This package implements exact HMC sampling for truncated multivariate gaussians with quadratic constraints.

## Installation
Within the desired installation environment run the following:
```bash
git clone git@github.com:erik-a-bensen/tmg_hmc.git
cd tmg_hmc 
pip install .
```

**Requirements:**
- Python 3.10+
- numpy
- scipy
- torch

## Quick Start

### Linearly Constrained Gaussian
To sample a 2d standard normal with y component restricted to be positive run the following:

```python
import numpy as np
from tmg_hmc import TMGSampler 

# Define the mean and covariance of the untruncated distribution
mu = np.zeros(2,1)
Sigma = np.identity(2)
sampler = TMGSampler(mu, Sigma)

# Define the constraint y >= 0
# Corresponds to A = 0, f = [0, 1], c = 0
f = np.array([0,1]).reshape(-1,1)
sampler.add_constraint(f=f)

# Sample 100 samples 
x0 = np.array([1,1]).reshape(-1,1) # Must satisfy our constraint
samples = sampler.sample(x0, n_samples=100, burn_in=100)
```

TODO: add plots and other examples.

## Documentation

- [Full API Reference](API_DOCS.md) - Complete documentation of all functions and classes

<!-- ## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change. -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Implemented based on the paper Exact Hamiltonian Monte Carlo by Pakman and Paninski (2014)

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