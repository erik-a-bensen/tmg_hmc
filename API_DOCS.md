# API Reference

> **Package:** `tmg_hmc` v0.1.1
>
> This package implements exact HMC sampling for truncated multivariate gaussians with quadratic constraints.

## `TMGSampler` (class)

```python
TMGSampler(mu: Array = None, Sigma: Array = None, T: float = np.pi / 2, gpu: bool = False, Sigma_half: Array = None)
```

Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions<br>
with linear and quadratic constraints.

**Constructor:**

Parameters<br>
----------<br>
mu : Array, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mean vector of the Gaussian distribution. If None, defaults to zero vector.<br>
Sigma : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Covariance matrix of the Gaussian distribution. Must be positive semi-definite.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Do not provide if Sigma_half is given.<br>
T : float, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Integration time for the Hamiltonian dynamics. Default is pi/2.<br>
gpu : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use GPU acceleration with PyTorch. Default is False.<br>
Sigma_half : Array, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Matrix such that Sigma_half @ Sigma_half.T = Sigma. <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;If provided, Sigma is not needed.

### Methods

### `add_constraint`

```python
add_constraint(A: Array = None, f: Array = None, c: float = 0.0, sparse: bool = True, compiled: bool = True) -> None
```

Adds a constraint to the sampler of the form:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;x.T @ A @ x + f.T @ x + c >= 0

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
The constraint is automatically transformed to account for the Gaussian's mean and covariance.<br>
The transformed constraint becomes:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;y.T @ (S @ A @ S) @ y + (2 * S @ A @ mu + S @ f).T @ y + (mu.T @ A @ mu + mu.T @ f + c) >= 0<br>
where y = S^{-1} (x - mu) and S = Sigma_half.<br>
Depending on whether A and f are non-zero, the appropriate constraint type is chosen.

### `load`

```python
load(filename: str) -> TMGSampler
```

Loads the sampler state from a pickled file.

### `sample`

```python
sample(x0: Array = None, n_samples: int = 100, burn_in: int = 100, verbose = False, cont: bool = False) -> Array
```

Generates samples from the truncated multivariate Gaussian distribution.

Parameters<br>
----------<br>
x0 : Array<br>
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

## `BaseQuadraticConstraint` (class)

```python
BaseQuadraticConstraint(args, kwargs)
```

Base class for quadratic constraints

### Methods

### `A_dot_x`

```python
A_dot_x(x: Array) -> Array
```

Compute A x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate A x at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Result of A x computation

### `compute_q`

```python
compute_q(a: Array, b: Array) -> Tuple[float, ...]
```

Compute the coefficients of the constraint equation along the trajectory defined by a and b

### `compute_q_`

```python
compute_q_(a: Array, b: Array) -> Tuple[float, ...]
```

Placeholder method for dense q term computation

### `compute_q_sparse`

```python
compute_q_sparse(a: Array, b: Array) -> Tuple[float, ...]
```

Placeholder method for sparse q term computation

### `deserialize`

```python
deserialize(d: dict, gpu: bool) -> Constraint
```

Deserialize the constraint from a dictionary

Parameters<br>
----------<br>
d : dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint<br>
gpu : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to load tensors onto the GPU

Returns<br>
-------<br>
Constraint<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deserialized constraint object

### `hit_time`

```python
hit_time(a: Array, b: Array) -> Array
```

Compute the hit time of the constraint along the trajectory defined by a and b

### `is_satisfied`

```python
is_satisfied(x: Array) -> bool
```

Check if the constraint is satisfied at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True if the constraint is satisfied, False otherwise

### `is_zero`

```python
is_zero(x: Array) -> Tuple[bool, bool]
```

Check if the constraint is zero at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
Tuple[bool, bool]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(is_strictly_zero, is_approximately_zero)

### `normal`

```python
normal(x: Array) -> Array
```

Compute the normal vector of the constraint at x

### `normal_`

```python
normal_(x: Array) -> Array
```

Placeholder method for dense normal vector computation

### `normal_sparse`

```python
normal_sparse(x: Array) -> Array
```

Placeholder method for sparse normal vector computation

### `reflect`

```python
reflect(x: Array, xdot: Array) -> Array
```

Reflect the velocity xdot at the constraint surface defined by x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point on the constraint surface<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Velocity to be reflected

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reflected velocity

### `serialize`

```python
serialize() -> dict
```

Serialize the constraint to a dictionary

Returns<br>
-------<br>
dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint

### `value`

```python
value(x: Array) -> float
```

Compute the value of the constraint at x

### `value_`

```python
value_(x: Array) -> float
```

Placeholder method for dense value computation

### `value_sparse`

```python
value_sparse(x: Array) -> float
```

Placeholder method for sparse value computation

### `x_dot_A_dot_x`

```python
x_dot_A_dot_x(x: Array) -> float
```

Compute x^T A x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate x^T A x at

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Result of x^T A x computation

## `Constraint` (class)

```python
Constraint(args, kwargs)
```

Abstract base class for constraints

### Methods

### `compute_q`

```python
compute_q(a: Array, b: Array) -> Tuple[float, ...]
```

Compute the coefficients of the constraint equation along the trajectory defined by a and b

### `deserialize`

```python
deserialize(d: dict, gpu: bool) -> Constraint
```

Deserialize the constraint from a dictionary

Parameters<br>
----------<br>
d : dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint<br>
gpu : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to load tensors onto the GPU

Returns<br>
-------<br>
Constraint<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deserialized constraint object

### `hit_time`

```python
hit_time(a: Array, b: Array) -> Array
```

Compute the hit time of the constraint along the trajectory defined by a and b

### `is_satisfied`

```python
is_satisfied(x: Array) -> bool
```

Check if the constraint is satisfied at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True if the constraint is satisfied, False otherwise

### `is_zero`

```python
is_zero(x: Array) -> Tuple[bool, bool]
```

Check if the constraint is zero at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
Tuple[bool, bool]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(is_strictly_zero, is_approximately_zero)

### `normal`

```python
normal(x: Array) -> Array
```

Compute the normal vector of the constraint at x

### `reflect`

```python
reflect(x: Array, xdot: Array) -> Array
```

Reflect the velocity xdot at the constraint surface defined by x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point on the constraint surface<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Velocity to be reflected

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reflected velocity

### `serialize`

```python
serialize() -> dict
```

Serialize the constraint to a dictionary

Returns<br>
-------<br>
dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint

### `value`

```python
value(x: Array) -> float
```

Compute the value of the constraint at x

## `LinearConstraint` (class)

```python
LinearConstraint(f: Array, c: float)
```

Constraint of the form fx + c >= 0

**Constructor:**

Parameters<br>
----------<br>
f : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Coefficient vector<br>
c : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Constant term

### Methods

### `compute_q`

```python
compute_q(a: Array, b: Array) -> Tuple[float, float]
```

Compute the 2 q terms for the linear constraint

Parameters<br>
----------<br>
a : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory<br>
b : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory

Returns<br>
-------<br>
Tuple[float, float]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;q terms for the constraint

Notes<br>
-----<br>
These expressions are defined such that Eqn 2.22 in Pakman and Paninski (2014) <br>
simplifies to: q1 sin(t) + q2 cos(t) + c = 0

### `deserialize`

```python
deserialize(d: dict, gpu: bool) -> Constraint
```

Deserialize the constraint from a dictionary

Parameters<br>
----------<br>
d : dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint<br>
gpu : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to load tensors onto the GPU

Returns<br>
-------<br>
Constraint<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deserialized constraint object

### `hit_time`

```python
hit_time(x: Array, xdot: Array) -> Array
```

Compute the hit time of the constraint along the trajectory defined by x and xdot

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Hit time of the constraint along the trajectory

Notes<br>
-----<br>
Hit time is computed by solving Eqn 2.26 in Pakman and Paninski (2014)<br>
See resources/HMC_exact_soln.nb for derivation<br>
Due to the sum of inverse trig functions, we check the solution and <br>
the solution +- pi to ensure we capture all hit times. 

Only positive hit times are returned and any ghost solutions are filtered <br>
out at a later stage.

### `is_satisfied`

```python
is_satisfied(x: Array) -> bool
```

Check if the constraint is satisfied at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True if the constraint is satisfied, False otherwise

### `is_zero`

```python
is_zero(x: Array) -> Tuple[bool, bool]
```

Check if the constraint is zero at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
Tuple[bool, bool]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(is_strictly_zero, is_approximately_zero)

### `normal`

```python
normal(x: Array) -> Array
```

Compute the normal vector of the constraint at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the normal vector at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Normal vector of the constraint at x given by f

### `reflect`

```python
reflect(x: Array, xdot: Array) -> Array
```

Reflect the velocity xdot at the constraint surface defined by x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point on the constraint surface<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Velocity to be reflected

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reflected velocity

### `serialize`

```python
serialize() -> dict
```

Serialize the constraint to a dictionary

Returns<br>
-------<br>
dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint

### `value`

```python
value(x: Array) -> float
```

Compute the value of the constraint at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Value of the constraint at x given by f^T x + c

## `QuadraticConstraint` (class)

```python
QuadraticConstraint(A: Array, b: Array, c: float, S: Array, sparse: bool = True, compiled: bool = True)
```

Constraint of the form x**T A x + b**T x + c >= 0

**Constructor:**

Parameters<br>
----------<br>
A : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The quadratic term matrix<br>
b : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The linear term vector<br>
c : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The constant term<br>
S : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The transformation matrix given by the Symmetric Sqrt of the Mass matrix<br>
sparse : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use sparse matrix computations, by default True<br>
compiled : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use compiled code, by default True

Notes<br>
-----<br>
If A is a sparse matrix, sparse computations are used regardless of the<br>
sparse parameter.<br>
It is highly recommended to use compiled code for performance reasons.

### Methods

### `A_dot_x`

```python
A_dot_x(x: Array) -> Array
```

Compute A x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate A x at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Result of A x computation

### `build_from_dict`

```python
build_from_dict(d: dict, gpu: bool) -> 'QuadraticConstraint'
```

Build a QuadraticConstraint from a dictionary representation

Parameters<br>
----------<br>
d : dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint<br>
gpu : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to load tensors onto the GPU

Returns<br>
-------<br>
QuadraticConstraint<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The constructed constraint

### `compute_q`

```python
compute_q(a: Array, b: Array) -> Tuple[float, ...]
```

Compute the coefficients of the constraint equation along the trajectory defined by a and b

### `compute_q_`

```python
compute_q_(a: Array, b: Array) -> Tuple[float, float, float, float, float]
```

Compute the 5 q terms for the quadratic constraint using dense matrix computations

Parameters<br>
----------<br>
a : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory<br>
b : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory

Returns<br>
-------<br>
Tuple[float, float, float, float, float]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;q terms for the quadratic constraint

Notes<br>
-----<br>
These expressions are defined in Eqns 2.40-2.44 in Pakman and Paninski (2014)

### `compute_q_sparse`

```python
compute_q_sparse(a: Array, b: Array) -> Tuple[float, float, float, float, float]
```

Compute the 5 q terms for the quadratic constraint using sparse matrix computations

Parameters<br>
----------<br>
a : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory<br>
b : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory

Returns<br>
-------<br>
Tuple[float, float, float, float, float]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;q terms for the quadratic constraint

Notes<br>
-----<br>
These expressions are defined in Eqns 2.40-2.44 in Pakman and Paninski (2014)

### `deserialize`

```python
deserialize(d: dict, gpu: bool) -> Constraint
```

Deserialize the constraint from a dictionary

Parameters<br>
----------<br>
d : dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint<br>
gpu : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to load tensors onto the GPU

Returns<br>
-------<br>
Constraint<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deserialized constraint object

### `hit_time`

```python
hit_time(a: Array, b: Array) -> Array
```

Compute the hit time of the constraint along the trajectory defined by a and b

### `hit_time_cpp`

```python
hit_time_cpp(x: Array, xdot: Array) -> Array
```

Compute the hit time for the quadratic constraint using compiled code

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The hit time for the constraint

Notes<br>
-----<br>
Hit time is computed by solving Eqn 2.48 in Pakman and Paninski (2014)<br>
See resources/HMC_exact_soln.nb for derivation<br>
Only positive hit times are returned and any ghost solutions are filtered <br>
out at a later stage.

Compiled code is both written in C++ and optimized to remove all redundant computations<br>
see paper for details.

### `hit_time_py`

```python
hit_time_py(x: Array, xdot: Array) -> Array
```

Compute the hit time for the quadratic constraint using Python code

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The hit time for the constraint

Notes<br>
-----<br>
Hit time is computed by solving Eqn 2.48 in Pakman and Paninski (2014)<br>
See resources/HMC_exact_soln.nb for derivation<br>
Only positive hit times are returned and any ghost solutions are filtered <br>
out at a later stage.

It is highly recommended to use the compiled version for performance reasons. <br>
This Python version is maintained for testing and validation purposes.

### `is_satisfied`

```python
is_satisfied(x: Array) -> bool
```

Check if the constraint is satisfied at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True if the constraint is satisfied, False otherwise

### `is_zero`

```python
is_zero(x: Array) -> Tuple[bool, bool]
```

Check if the constraint is zero at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
Tuple[bool, bool]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(is_strictly_zero, is_approximately_zero)

### `normal`

```python
normal(x: Array) -> Array
```

Compute the normal vector of the constraint at x

### `normal_`

```python
normal_(x: Array) -> Array
```

Compute the normal vector at x using dense matrix computations<br>
Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the normal vector at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Normal vector at x given by 2 * A @ x + b

### `normal_sparse`

```python
normal_sparse(x: Array) -> Array
```

Compute the normal vector at x using sparse matrix computations<br>
Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the normal vector at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Normal vector at x given by 2 * A @ x + b

### `reflect`

```python
reflect(x: Array, xdot: Array) -> Array
```

Reflect the velocity xdot at the constraint surface defined by x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point on the constraint surface<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Velocity to be reflected

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reflected velocity

### `serialize`

```python
serialize() -> dict
```

Serialize the constraint to a dictionary

Returns<br>
-------<br>
dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint

### `value`

```python
value(x: Array) -> float
```

Compute the value of the constraint at x

### `value_`

```python
value_(x: Array) -> float
```

Compute the value of the constraint at x using dense matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The value of the constraint at x given by x^T A x + b^T x + c

### `value_sparse`

```python
value_sparse(x: Array) -> float
```

Compute the value of the constraint at x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The value of the constraint at x given by x^T A x + b^T x + c

### `x_dot_A_dot_x`

```python
x_dot_A_dot_x(x: Array) -> float
```

Compute x^T A x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate x^T A x at

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Result of x^T A x computation

## `SimpleQuadraticConstraint` (class)

```python
SimpleQuadraticConstraint(A: Array, c: float, S: Array, sparse: bool = False)
```

Constraint of the form x^T A x + c >= 0

**Constructor:**

Parameters<br>
----------<br>
A : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Quadratic coefficient matrix<br>
c : float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Constant term<br>
S : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Transformation matrix given by the Symmetric Sqrt of the Mass matrix<br>
sparse : bool, optional<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to use sparse matrix computations, by default False

Notes<br>
-----<br>
If A is a sparse matrix, sparse computations are used regardless of the<br>
sparse parameter.

### Methods

### `A_dot_x`

```python
A_dot_x(x: Array) -> Array
```

Compute A x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate A x at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Result of A x computation

### `build_from_dict`

```python
build_from_dict(d: dict, gpu: bool) -> SimpleQuadraticConstraint
```

Build a SimpleQuadraticConstraint from a dictionary representation

Parameters<br>
----------<br>
d : dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint<br>
gpu : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to load tensors onto the GPU

Returns<br>
-------<br>
SimpleQuadraticConstraint<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The constructed constraint

### `compute_q`

```python
compute_q(a: Array, b: Array) -> Tuple[float, ...]
```

Compute the coefficients of the constraint equation along the trajectory defined by a and b

### `compute_q_`

```python
compute_q_(a: Array, b: Array) -> Tuple[float, float, float]
```

Compute the 3 q terms for the simple quadratic constraint using dense matrix computations

Parameters<br>
----------<br>
a : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory<br>
b : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory

Returns<br>
-------<br>
Tuple[float, float, float]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;q terms for the constraint

Notes<br>
-----<br>
These expressions are the nonzero q terms defined in equation 2.45 in Pakman and Paninski (2014)

### `compute_q_sparse`

```python
compute_q_sparse(a: Array, b: Array) -> Tuple[float, float, float]
```

Compute the 3 q terms for the simple quadratic constraint using sparse matrix computations

Parameters<br>
----------<br>
a : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory<br>
b : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory

Returns<br>
-------<br>
Tuple[float, float, float]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;q terms for the constraint

Notes<br>
-----<br>
These expressions are the nonzero q terms defined in equation 2.45 in Pakman and Paninski (2014)

### `deserialize`

```python
deserialize(d: dict, gpu: bool) -> Constraint
```

Deserialize the constraint from a dictionary

Parameters<br>
----------<br>
d : dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint<br>
gpu : bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Whether to load tensors onto the GPU

Returns<br>
-------<br>
Constraint<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Deserialized constraint object

### `hit_time`

```python
hit_time(x: Array, xdot: Array) -> Array
```

Compute the hit time for the simple quadratic constraint

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The position of the point in the HMC trajectory<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The velocity of the point in the HMC trajectory

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The hit time for the constraint

Notes<br>
-----<br>
Hit time is computed by solving Eqn 2.45 in Pakman and Paninski (2014)<br>
See resources/HMC_exact_soln.nb for derivation<br>
Only positive hit times are returned and any ghost solutions are filtered <br>
out at a later stage.

### `is_satisfied`

```python
is_satisfied(x: Array) -> bool
```

Check if the constraint is satisfied at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
bool<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;True if the constraint is satisfied, False otherwise

### `is_zero`

```python
is_zero(x: Array) -> Tuple[bool, bool]
```

Check if the constraint is zero at x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at<br>
Returns<br>
-------<br>
Tuple[bool, bool]<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(is_strictly_zero, is_approximately_zero)

### `normal`

```python
normal(x: Array) -> Array
```

Compute the normal vector of the constraint at x

### `normal_`

```python
normal_(x: Array) -> Array
```

Compute the normal vector at x using dense matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the normal vector at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Normal vector at x given by 2 * A @ x

### `normal_sparse`

```python
normal_sparse(x: Array) -> Array
```

Compute the normal vector at x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the normal vector at

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Normal vector at x given by 2 * A @ x

### `reflect`

```python
reflect(x: Array, xdot: Array) -> Array
```

Reflect the velocity xdot at the constraint surface defined by x

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point on the constraint surface<br>
xdot : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Velocity to be reflected

Returns<br>
-------<br>
Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Reflected velocity

### `serialize`

```python
serialize() -> dict
```

Serialize the constraint to a dictionary

Returns<br>
-------<br>
dict<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dictionary representation of the constraint

### `value`

```python
value(x: Array) -> float
```

Compute the value of the constraint at x

### `value_`

```python
value_(x: Array) -> float
```

Compute the value of the constraint at x using dense matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Value of the constraint at x given by x^T A x + c

### `value_sparse`

```python
value_sparse(x: Array) -> float
```

Compute the value of the constraint at x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate the constraint at

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Value of the constraint at x given by x^T A x + c

### `x_dot_A_dot_x`

```python
x_dot_A_dot_x(x: Array) -> float
```

Compute x^T A x using sparse matrix computations

Parameters<br>
----------<br>
x : Array<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Point to evaluate x^T A x at

Returns<br>
-------<br>
float<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Result of x^T A x computation

## `Tensor` (class)

```python
Tensor(args, kwargs)
```

Initialize self.  See help(type(self)) for accurate signature.

---

# Utils Module

## `Tensor` (class)

```python
Tensor(args, kwargs)
```

Initialize self.  See help(type(self)) for accurate signature.

## `arccos`

```python
arccos(x: float) -> <class 'float'>
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

Notes<br>
-----<br>
Uses the cmath.acos function to handle complex values and returns the real part.<br>
Can potentially create ghost values if the input is outside the range [-1, 1]. <br>
However, due to the complexity of the solution expressions this is necessary for <br>
numerical stability and ghost solutions are filtered out later.

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
get_sparse_elements(A: numpy.ndarray | tmg_hmc.utils.Tensor | scipy.sparse._coo.coo_matrix | None) -> Tuple[numpy.ndarray | tmg_hmc.utils.Tensor | scipy.sparse._coo.coo_matrix | None, numpy.ndarray | tmg_hmc.utils.Tensor | scipy.sparse._coo.coo_matrix | None, numpy.ndarray | tmg_hmc.utils.Tensor | scipy.sparse._coo.coo_matrix | None]
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

## `sparsify`

```python
sparsify(A: numpy.ndarray | tmg_hmc.utils.Tensor | scipy.sparse._coo.coo_matrix | None) -> numpy.ndarray | tmg_hmc.utils.Tensor | scipy.sparse._coo.coo_matrix | None
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

## `to_scalar`

```python
to_scalar(x: numpy.ndarray | tmg_hmc.utils.Tensor | scipy.sparse._coo.coo_matrix | None | float) -> <class 'float'>
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

calc_all_solutions(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> numpy.typing.NDArray[numpy.float64]


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

soln1(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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

soln2(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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

soln3(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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

soln4(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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

soln5(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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

soln6(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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

soln7(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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

soln8(arg0: typing.SupportsFloat, arg1: typing.SupportsFloat, arg2: typing.SupportsFloat, arg3: typing.SupportsFloat, arg4: typing.SupportsFloat) -> float


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
given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using <br>
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
