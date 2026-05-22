tmg-hmc documentation
======================

**tmg-hmc** is a Python package for exact Hamiltonian Monte Carlo sampling from 
truncated multivariate Gaussian distributions with linear and quadratic constraints.

It implements the algorithm of `Pakman & Paninski (2014) 
<https://doi.org/10.1080/10618600.2013.788448>`_ with support for:

- Linear constraints :math:`\mathbf{f}^\top \mathbf{x} + c \geq 0`
- Simple quadratic constraints :math:`\mathbf{x}^\top A \mathbf{x} + c \geq 0`
- Full quadratic constraints :math:`\mathbf{x}^\top A \mathbf{x} + \mathbf{f}^\top \mathbf{x} + c \geq 0`
- Product constraints :math:`\prod_i \left(\mathbf{x}^\top A_i \mathbf{x} + \mathbf{f}_i^\top \mathbf{x} + c_i\right) \geq 0`
- Optional GPU acceleration via PyTorch

Installation
------------

.. code-block:: bash

   pip install tmg-hmc

For GPU support:

.. code-block:: bash

   pip install tmg-hmc[gpu]

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   constraints
   api