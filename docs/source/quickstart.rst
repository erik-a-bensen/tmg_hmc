Quick Start
===========

Linearly Constrained Gaussian
------------------------------

Sample a 2D standard normal with the y-component restricted to be positive:

.. code-block:: python

   import numpy as np
   from tmg_hmc import TMGSampler

   # Define the mean and covariance of the untruncated distribution
   mu = np.zeros((2, 1))
   Sigma = np.identity(2)
   sampler = TMGSampler(mu, Sigma)

   # Define the constraint y >= 0
   # This corresponds to: f^T x + c >= 0 where f = [0, 1] and c = 0
   f = np.array([[0], [1]])
   sampler.add_constraint(f=f, c=0)

   # Sample 100 samples with 100 burn-in iterations
   x0 = np.array([[1], [1]])  # Initial point (must satisfy constraints)
   samples = sampler.sample(x0, n_samples=100, burn_in=100)

Quadratically Constrained Gaussian
------------------------------------

Sample from a Gaussian constrained to a circular region:

.. code-block:: python

   import numpy as np
   from tmg_hmc import TMGSampler

   mu = np.zeros((2, 1))
   Sigma = np.identity(2)
   sampler = TMGSampler(mu, Sigma)

   # Constrain to inside a circle of radius 2: x^2 + y^2 <= 4
   # Expressed as: -x^T A x + c >= 0 with A = I, c = 4
   A = -np.identity(2)
   c = 4
   sampler.add_constraint(A=A, c=c)

   x0 = np.array([[0.5], [0.5]])
   samples = sampler.sample(x0, n_samples=1000, burn_in=100)

Multiple Constraints
---------------------

Combine multiple constraints to define a box region:

.. code-block:: python

   import numpy as np
   from tmg_hmc import TMGSampler

   mu = np.zeros((2, 1))
   Sigma = np.identity(2)
   sampler = TMGSampler(mu, Sigma)

   # Box constraint: -1 <= x, y <= 1
   sampler.add_constraint(f=np.array([[ 1], [0]]), c=1)  # x >= -1
   sampler.add_constraint(f=np.array([[-1], [0]]), c=1)  # x <= 1
   sampler.add_constraint(f=np.array([[0], [ 1]]), c=1)  # y >= -1
   sampler.add_constraint(f=np.array([[0], [-1]]), c=1)  # y <= 1

   x0 = np.array([[0], [0]])
   samples = sampler.sample(x0, n_samples=1000, burn_in=100)