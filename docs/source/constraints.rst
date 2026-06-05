Constraint Construction
=======================

This page explains how to construct constraints for the :class:`~tmg_hmc.TMGSampler`.
All constraints take the general form:

.. math::

    \mathbf{x}^T A \mathbf{x} + \mathbf{f}^T \mathbf{x} + c \geq 0

where :math:`A` is a symmetric matrix, :math:`\mathbf{f}` is a vector, and :math:`c` is a scalar.
Depending on which terms are non-zero, the sampler uses one of three constraint types.

The Transformation
------------------

When a constraint is added, the sampler automatically transforms it to the standardized
space :math:`\mathbf{y} = S^{-1}(\mathbf{x} - \pmb{\mu})` where :math:`S = \Sigma^{1/2}`.
You do not need to apply this transformation yourself. The transformed constraint becomes:

.. math::

    \mathbf{y}^T \tilde{A} \mathbf{y} + \tilde{\mathbf{f}}^T \mathbf{y} + \tilde{c} \geq 0

where:

.. math::

    \tilde{A} = S A S, \quad
    \tilde{\mathbf{f}} = 2 S A \pmb{\mu} + S \mathbf{f}, \quad
    \tilde{c} = \pmb{\mu}^T A \pmb{\mu} + \mathbf{f}^T \pmb{\mu} + c

The constraint type is inferred based on which terms are non-zero after this transformation:

- :math:`\tilde{A} = 0`: :class:`~tmg_hmc.constraints.LinearConstraint`
- :math:`\tilde{A} \neq 0` and :math:`\tilde{\mathbf{f}} = 0`: :class:`~tmg_hmc.constraints.SimpleQuadraticConstraint`
- :math:`\tilde{A} \neq 0` and :math:`\tilde{\mathbf{f}} \neq 0`: :class:`~tmg_hmc.constraints.QuadraticConstraint`

Since :math:`\tilde{\mathbf{f}} = 2 S A \pmb{\mu} + S \mathbf{f}`, a linear input constraint
(:math:`A = 0`) always remains linear after transformation because :math:`\tilde{A} = 0`
regardless of :math:`\pmb{\mu}`. However, a simple quadratic input constraint
(:math:`\mathbf{f} = 0`) may become a full quadratic constraint after transformation
if :math:`\pmb{\mu} \neq 0`, since the term :math:`2 S A \pmb{\mu}` introduces a
non-zero linear component :math:`\tilde{\mathbf{f}}`.

From Constraint to A, f, c
--------------------------

The key step in using the sampler is expressing your constraint in the standard form
:math:`\mathbf{x}^T A \mathbf{x} + \mathbf{f}^T \mathbf{x} + c \geq 0`. The following
examples walk through how to do this for common constraint types.

**Half-space:** :math:`x_2 \geq 0`

Rearrange to :math:`x_2 \geq 0`, which is already in the form :math:`\mathbf{f}^T \mathbf{x} + c \geq 0`:

.. math::

    A = 0, \quad \mathbf{f} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}, \quad c = 0

**Bounded component:** :math:`|x_1| \leq 2`

This is equivalent to :math:`-x_1^2 + 4 \geq 0`, giving:

.. math::

    A = \begin{pmatrix} -1 & 0 \\ 0 & 0 \end{pmatrix}, \quad \mathbf{f} = 0, \quad c = 4

**Inside an ellipse:** :math:`x_1^2 + x_1 x_2 + x_2^2 \leq 1`

Rearrange to :math:`-(x_1^2 + x_1 x_2 + x_2^2) + 1 \geq 0`. The quadratic form
:math:`x_1^2 + x_1 x_2 + x_2^2 = \mathbf{x}^T A \mathbf{x}` with :math:`A_{11} = 1`,
:math:`A_{22} = 1`, and :math:`A_{12} = A_{21} = 0.5` (note the off-diagonal entries
are half the coefficient of the cross term):

.. math::

    A = -\begin{pmatrix} 1 & 0.5 \\ 0.5 & 1 \end{pmatrix}, \quad \mathbf{f} = 0, \quad c = 1

**Below a parabola:** :math:`x_2 \leq x_1^2 + 0.5`

Rearrange to :math:`x_1^2 - x_2 + 0.5 \geq 0`:

.. math::

    A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \quad \mathbf{f} = \begin{pmatrix} 0 \\ -1 \end{pmatrix}, \quad c = 0.5

**Between two hyperplanes:** :math:`1 \leq \mathbf{a}^T \mathbf{x} \leq 3`

Split into two linear constraints:

.. math::

    \mathbf{f} = \mathbf{a}, \quad c = -1 \quad \text{(lower bound)}

.. math::

    \mathbf{f} = -\mathbf{a}, \quad c = 3 \quad \text{(upper bound)}

Linear Constraints
------------------

A linear constraint has the form :math:`\mathbf{f}^T \mathbf{x} + c \geq 0`.
Provide only ``f`` and ``c`` to :meth:`~tmg_hmc.TMGSampler.add_constraint`:

.. code-block:: python

    import numpy as np
    from tmg_hmc import TMGSampler

    mu = np.zeros((2, 1))
    Sigma = np.eye(2)
    sampler = TMGSampler(mu, Sigma)

    # Constrain x2 >= 0: f = [0, 1], c = 0
    sampler.add_constraint(f=np.array([[0.0], [1.0]]), c=0)

Multiple linear constraints can be combined to form box constraints:

.. code-block:: python

    # Box constraint: -1 <= x1 <= 1
    sampler.add_constraint(f=np.array([[ 1.0], [0.0]]), c=1)   # x1 >= -1
    sampler.add_constraint(f=np.array([[-1.0], [0.0]]), c=1)   # x1 <= 1

Simple Quadratic Constraints
----------------------------

A simple quadratic constraint has the form :math:`\mathbf{x}^T A \mathbf{x} + c \geq 0`
with no linear term. Provide only ``A`` and ``c``:

.. code-block:: python

    # Constrain to inside a unit circle: -x1^2 - x2^2 + 1 >= 0
    A = -np.eye(2)
    sampler.add_constraint(A=A, c=1)

    # Constrain to outside an ellipse: x1^2 + x1*x2 + x2^2 - 1 >= 0
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    sampler.add_constraint(A=A, c=-1)

.. note::

    ``A`` must be symmetric. The sampler will raise a ``ValueError`` if it is not.
    If the untruncated mean :math:`\pmb{\mu} \neq 0`, the transformed constraint may
    become a full quadratic constraint internally even if no ``f`` is provided.

Full Quadratic Constraints
--------------------------

A full quadratic constraint has the form
:math:`\mathbf{x}^T A \mathbf{x} + \mathbf{f}^T \mathbf{x} + c \geq 0`
with both ``A`` and ``f`` non-zero:

.. code-block:: python

    # Constrain x2 <= x1^2 + 0.5: x1^2 - x2 + 0.5 >= 0
    A = np.array([[1.0, 0.0], [0.0, 0.0]])
    f = np.array([[0.0], [-1.0]])
    sampler.add_constraint(A=A, f=f, c=0.5)

Product Constraints
-------------------

Product constraints allow sampling from non-convex regions defined by the product
of multiple linear or quadratic constraints:

.. math::

    \prod_{i=1}^k \left( \mathbf{x}^T A_i \mathbf{x} + \mathbf{f}_i^T \mathbf{x} + c_i \right) \geq 0

The product is satisfied when an even number of the individual factors are negative
(or all are non-negative). This enables sampling from regions that cannot be
expressed as a single convex constraint.

This is a capability unique to exact HMC. Because the algorithm computes exact
bounce times analytically, it can handle the complex boundary structure of product
constraints naturally. Standard HMC methods require gradient information and struggle
with such non-convex regions.

Product constraints are added via :meth:`~tmg_hmc.TMGSampler.add_product_constraint`,
which takes a list of parameter dictionaries:

.. code-block:: python

    # Sample from the region where x1*x2 is in [-1, 1]:
    # (x1*x2 + 1)(-x1*x2 + 1) >= 0
    parameters = [
        {"A": np.array([[0.0, 0.5], [0.5, 0.0]]), "c": 1.0},   # x1*x2 + 1 >= 0
        {"A": np.array([[0.0, -0.5], [-0.5, 0.0]]), "c": 1.0},  # -x1*x2 + 1 >= 0
    ]
    sampler.add_product_constraint(parameters=parameters)

The ``sparse`` and ``compiled`` Options
---------------------------------------

Both :meth:`~tmg_hmc.TMGSampler.add_constraint` and
:meth:`~tmg_hmc.TMGSampler.add_product_constraint` accept two optional flags:

``sparse`` (default: ``True``)
    If ``True``, the constraint matrices ``A`` and ``f`` are stored in sparse format.
    This is beneficial when the matrices are large and have many zero entries, as is
    common in Gaussian process applications. For small dense matrices it makes little
    difference.

``compiled`` (default: ``True``)
    If ``True``, uses the compiled C++ extension for computing hit times in full
    quadratic constraints. This is significantly faster than the pure Python
    implementation and should be left as ``True`` unless you are debugging or
    the compiled extension is unavailable.

Combining Constraints
---------------------

Multiple constraints can be added with repeated calls to
:meth:`~tmg_hmc.TMGSampler.add_constraint`. The sampler requires all constraints
to be satisfied simultaneously, truncating the Gaussian to the intersection of the
constraint regions:

.. code-block:: python

    mu = np.zeros((2, 1))
    Sigma = 2 * np.array([[1.0, 0.6], [0.6, 1.0]])
    sampler = TMGSampler(mu, Sigma)

    sampler.add_constraint(
        A=np.array([[1.0, 0], [0, 0]]),
        f=np.array([[0.0], [-1.0]]),
        c=1.5,
    )  # x2 <= x1^2 + 1.5
    sampler.add_constraint(A=-np.array([[1.0, 0], [0, 0]]), c=4)  # |x1| <= 2
    sampler.add_constraint(f=np.array([[0.0], [1.0]]), c=2)       # x2 >= -2

Initial Point
-------------

The initial point ``x0`` passed to :meth:`~tmg_hmc.TMGSampler.sample` must satisfy
all constraints. If it does not, a ``ValueError`` is raised. Choose a point that is
clearly in the interior of the feasible region:

.. code-block:: python

    x0 = np.array([[0.0], [0.0]])  # must satisfy all constraints
    samples = sampler.sample(x0, n_samples=1000, burn_in=100)