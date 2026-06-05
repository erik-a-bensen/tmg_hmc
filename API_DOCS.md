# API Reference

> **Package:** `tmg_hmc` v1.0.3
>
> This package implements exact HMC sampling for truncated multivariate gaussians with quadratic constraints.

## `TMGSampler` (class)

```python
TMGSampler(mu: Array | None = None, Sigma: Array | None = None, T: float | None = None, gpu: bool = False, Sigma_half: Array | None = None)
```

Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions<br>
with linear and quadratic constraints.

**Constructor:**

Parameters<br>
----------<br>
mu : Array, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean vector of the Gaussian distribution. If None, defaults to zero vector.<br>
Sigma : Array, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Covariance matrix of the Gaussian distribution. Must be positive semi-definite.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Do not provide if Sigma_half is given.<br>
T : float, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Integration time for the Hamiltonian dynamics. Default is :math:`\pi/2`.<br>
gpu : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use GPU acceleration with PyTorch. Default is False.<br>
Sigma_half : Array, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Matrix such that :math:`S S^T = \Sigma`.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If provided, Sigma is not needed.

### Methods

### `add_constraint`

```python
add_constraint(A: Array | None = None, f: Array | None = None, c: float = 0.0, sparse: bool = True, compiled: bool = True) -> None
```

Adds a constraint to the sampler.

.. math::

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\mathbf{x}^T A \mathbf{x} + \mathbf{f}^T \mathbf{x} + c \geq 0

Parameters<br>
----------<br>
A : Array, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Quadratic term matrix, defaults to the zero matrix if not provided.<br>
f : Array, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Linear term vector, defaults to the zero vector if not provided.<br>
c : float, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Constant term. Default is 0.0.<br>
sparse : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to store A and f in sparse format. Default is True.<br>
compiled : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use compiled constraint solutions for full quadratic constraints. Default is True.

Raises<br>
------<br>
ValueError<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If A is not symmetric when provided, or if neither A nor f is provided.

Notes<br>
-----<br>
The constraint is automatically transformed to account for the Gaussian's mean and covariance.

.. math::

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\mathbf{y}^T (S A S) \mathbf{y} + (2 S A \pmb{\mu} + S \mathbf{f})^T \mathbf{y} + (\pmb{\mu}^T A \pmb{\mu} + \pmb{\mu}^T \mathbf{f} + c) \geq 0

where :math:`\mathbf{y} = S^{-1}(\mathbf{x} - \pmb{\mu})` and :math:`S = \Sigma^{1/2}`.<br>
Depending on whether :math:`A` and :math:`\mathbf{f}` are non-zero after transformation,<br>
the appropriate constraint type is chosen.

### `add_product_constraint`

```python
add_product_constraint(parameters: list[list[Array]] | list[dict[str, Array]], sparse: bool = True, compiled: bool = True) -> None
```

Adds a constraint to the sampler.

.. math::

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\mathbf{x}^T A \mathbf{x} + \mathbf{f}^T \mathbf{x} + c \geq 0

Parameters<br>
----------<br>
parameters: list[list[Array]] | list[dict[str,Array]]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;List of constraint parameters as either lists [A, f, c] or dictionaries {'A': A, 'f': f, 'c': c}.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If list, each element must be of length 3 corresponding to A, f, and c.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If dictionary, missing keys 'A', 'f', and 'c' default to None, None, and 0.0 respectively.<br>
sparse : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to store A and f in sparse format. Default is True.<br>
compiled : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use compiled constraint solutions for full quadratic constraints. Default is True.

Raises<br>
------<br>
ValueError<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If A is not symmetric when provided, or if neither A nor f is provided.

Notes<br>
-----<br>
For product constraints, you must provide lists of each component (A, f, c).<br>
The constraint is automatically transformed to account for the Gaussian's mean and covariance.

.. math::

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;\mathbf{y}^T (S A S) \mathbf{y} + (2 S A \pmb{\mu} + S \mathbf{f})^T \mathbf{y} + (\pmb{\mu}^T A \pmb{\mu} + \pmb{\mu}^T \mathbf{f} + c) \geq 0

where :math:`\mathbf{y} = S^{-1}(\mathbf{x} - \pmb{\mu})` and :math:`S = \Sigma^{1/2}`.<br>
Depending on whether :math:`A` and :math:`\mathbf{f}` are non-zero after transformation,<br>
the appropriate constraint type is chosen.

### `load`

```python
load(filename: str) -> TMGSampler
```

Loads the sampler state from a pickled file.

### `sample`

```python
sample(x0: Array | None = None, n_samples: int = 100, burn_in: int = 100, verbose = False, cont: bool = False) -> Array
```

Generates samples from the truncated multivariate Gaussian distribution.

Parameters<br>
----------<br>
x0 : Array | None<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Initial point for the sampler. Optional if cont is True.<br>
n_samples : int, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of samples to generate, default is 100.<br>
burn_in : int, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Number of burn-in iterations, default is 100.<br>
verbose : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to print verbose output, default is False.<br>
cont : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to continue from the last sampled point. Default is False.

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generated samples.

Raises<br>
------<br>
ValueError<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If cont is False and x0 is not provided, or if x0 does not satisfy constraints.

### `sample_xdot`

```python
sample_xdot() -> Array
```

Samples a new momentum vector xdot from the standard normal distribution handling GPU if necessary.

### `save`

```python
save(filename: str) -> None
```

Saves the sampler state to a pickled file.

---

# Constraints Module

---

# Utils Module

## `arccos`

```python
arccos(x: complex) -> <class 'float'>
```

Computes the real component of the arccosine of a value.

Parameters<br>
----------<br>
x : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The input value.

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The real component of the arccosine of the input value.

## `compiled_library_available`

```python
compiled_library_available() -> <class 'bool'>
```

Checks if the compiled shared library is available.

Return<br>
-------<br>
bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True if the shared library is available, False otherwise.

## `get_sparse_elements`

```python
get_sparse_elements(A: numpy.ndarray | tmg_hmc.gpu_utils.Tensor) -> Tuple[numpy.ndarray | tmg_hmc.gpu_utils.Tensor, numpy.ndarray | tmg_hmc.gpu_utils.Tensor, numpy.ndarray | tmg_hmc.gpu_utils.Tensor]
```

Extracts the row, column, and data elements from a sparse matrix.

Parameters<br>
----------<br>
A : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The input sparse matrix.

Returns<br>
-------<br>
Tuple[Array, Array, Array]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A tuple containing the row indices, column indices, and data values of the sparse matrix.

## `is_nonzero_array`

```python
is_nonzero_array(x: numpy.ndarray | tmg_hmc.gpu_utils.Tensor) -> <class 'bool'>
```

Checks if the input array is non-zero.

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The input array to be checked.

Returns<br>
-------<br>
bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True if the array is non-zero, False otherwise.

## `sparsify`

```python
sparsify(A: numpy.ndarray | tmg_hmc.gpu_utils.Tensor) -> scipy.sparse._csc.csc_matrix | scipy.sparse._csr.csr_matrix | scipy.sparse._coo.coo_matrix | tmg_hmc.gpu_utils.Tensor
```

Converts a dense numpy array or a PyTorch tensor to a sparse COO matrix.

Parameters<br>
----------<br>
A : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The input array to be converted to a sparse matrix.

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The sparse COO matrix representation of the input array.

## `stable_acos`

```python
stable_acos(x: complex) -> <class 'complex'>
```

Computes a numerically stable arccosine for complex numbers.

Parameters<br>
----------<br>
x : complex<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The input complex number.

Returns<br>
-------<br>
complex<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The arccosine of the input complex number.

## `to_scalar`

```python
to_scalar(x: numpy.ndarray | tmg_hmc.gpu_utils.Tensor | float) -> <class 'float'>
```

Converts a scalar array or a float to a float.

Parameters<br>
----------<br>
x : Array | float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The input value to be converted.

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The converted float value.

---

# Compiled Module

## `calc_all_solutions`

```python
calc_all_solutions(...)
```

calc_all_solutions(arg0: typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsFloat | typing.SupportsIndex) -> numpy.typing.NDArray[numpy.float64]


calc_all_solutions(q1: float, q2: float, q3: float, q4: float, q5: float) -> np.ndarray

Compute all 8 solutions for the full quadratic constraint hit time.

This function computes all eight possible hit times for the quadratic constraint<br>
used in Hamiltonian Monte Carlo sampling, following the derivations in Pakman<br>
and Paninski (2014).

Parameters<br>
----------<br>
q1 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).<br>
q2 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The second parameter defined in Eqn 2.41.<br>
q3 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The third parameter defined in Eqn 2.42.<br>
q4 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The fourth parameter defined in Eqn 2.43.<br>
q5 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The fifth parameter defined in Eqn 2.44.

Returns<br>
-------<br>
numpy.ndarray<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A 1D NumPy array of length 8 containing all computed solutions.

Notes<br>
-----<br>
The solutions are derived from the quartic equation associated with the<br>
quadratic constraint hit time (Eqns 2.48–2.53 in the reference). These expressions<br>
were computed symbolically in Mathematica, exported to C, and then optimized<br>
for performance by eliminating redundant calculations.

Memory management is handled automatically.

## `soln1`

```python
soln1(...)
```

soln1(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln1(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the first solution for the quadratic constraint hit time.

Parameters<br>
----------<br>
See `calc_all_solutions` for parameter descriptions.

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The computed first solution.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
The solution is derived from the quartic equation associated with the quadratic constraint hit time<br>
given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using<br>
Mathematica and then exported to C. See resources/HMC_exact_soln.nb for details.

It is not recommended to use this function directly due to its complexity. Instead, use 'calc_all_solutions'<br>
which has been optimized to remove redundant calculations.<br>
This function is maintained for reference and validation purposes.

## `soln2`

```python
soln2(...)
```

soln2(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln2(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the second solution for the quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See 'soln1' for details. 

## `soln3`

```python
soln3(...)
```

soln3(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln3(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the third solution for the quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See 'soln1' for details. 

## `soln4`

```python
soln4(...)
```

soln4(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln4(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the fourth solution for the quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See 'soln1' for details. 

## `soln5`

```python
soln5(...)
```

soln5(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln5(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the fifth solution for the quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See 'soln1' for details. 

## `soln6`

```python
soln6(...)
```

soln6(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln6(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the sixth solution for the quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See 'soln1' for details. 

## `soln7`

```python
soln7(...)
```

soln7(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln7(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the seventh solution for the quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See 'soln1' for details. 

## `soln8`

```python
soln8(...)
```

soln8(arg0: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg1: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg2: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg3: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex, arg4: typing.SupportsComplex | typing.SupportsFloat | typing.SupportsIndex) -> float


soln8(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

Compute the eighth solution for the quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See 'soln1' for details. 

---

# Quad Solns Module

## `soln1`

```python
soln1(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the first of the 8 solutions for the full quadratic constraint hit time.

Parameters<br>
----------<br>
q1 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).<br>
q2 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).<br>
q3 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).<br>
q4 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).<br>
q5 : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The computed solution.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
The solution is derived from the quartic equation associated with the quadratic constraint hit time<br>
given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using<br>
Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.<br>
See resources/HMC_exact_soln.nb for details.

It is not recommended to use this function directly due to its complexity and slow performance. Instead,<br>
use the compiled module for efficient computation of all solutions.<br>
This function is maintained for reference and validation purposes.

## `soln2`

```python
soln2(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the second of the 8 solutions for the full quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See soln1 for details.

## `soln3`

```python
soln3(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the third of the 8 solutions for the full quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See soln1 for details.

## `soln4`

```python
soln4(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the fourth of the 8 solutions for the full quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See soln1 for details.

## `soln5`

```python
soln5(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the fifth of the 8 solutions for the full quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See soln1 for details.

## `soln6`

```python
soln6(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the sixth of the 8 solutions for the full quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See soln1 for details.

## `soln7`

```python
soln7(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the seventh of the 8 solutions for the full quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See soln1 for details.

## `soln8`

```python
soln8(q1: float, q2: float, q3: float, q4: float, q5: float) -> <class 'float'>
```

Computes the eighth of the 8 solutions for the full quadratic constraint hit time.

Notes<br>
-----<br>
DO NOT MODIFY THIS FUNCTION<br>
See soln1 for details.
