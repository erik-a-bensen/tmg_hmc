# TMG HMC API Documentation

**Package:** `tmg_hmc` v0.0.2

**Description:** This package implements exact HMC sampling for truncated multivariate gaussians with quadratic constraints.

---

## Table of Contents

- [tmg_hmc (main)](#tmg_hmc)
- [constraints](#tmg_hmcconstraints)
- [utils](#tmg_hmcutils)
- [quadratic_solns](#tmg_hmcquadratic_solns)

---

## tmg_hmc (main)

### Class: `TMGSampler`


    Hamiltonian Monte Carlo sampler for Multivariate Gaussian distributions
    with linear and quadratic constraints.
    

#### `__init__(mu=None, Sigma=None, T=1.5707963267948966, gpu=False, Sigma_half=None)`


        Parameters
        ----------
        mu : Array, optional
            Mean vector of the Gaussian distribution. If None, defaults to zero vector.
        Sigma : Array
            Covariance matrix of the Gaussian distribution. Must be positive semi-definite.
            Do not provide if Sigma_half is given.
        T : float, optional
            Integration time for the Hamiltonian dynamics. Default is pi/2.
        gpu : bool, optional
            Whether to use GPU acceleration with PyTorch. Default is False.
        Sigma_half : Array, optional
            Matrix such that Sigma_half @ Sigma_half.T = Sigma. 
            If provided, Sigma is not needed.
        

#### `add_constraint(A=None, f=None, c=0.0, sparse=True, compiled=True)`


        Adds a constraint to the sampler of the form:
            x.T @ A @ x + f.T @ x + c >= 0

        Parameters
        ----------
        A : Array, optional
            Quadratic term matrix, defaults to the zero matrix if not provided.
        f : Array, optional
            Linear term vector, defaults to the zero vector if not provided.
        c : float, optional
            Constant term. Default is 0.0.
        sparse : bool, optional
            Whether to store A and f in sparse format. Default is True.
        compiled : bool, optional
            Whether to use compiled constraint solutions for full quadratic constraints. Default is True.

        Raises
        ------
        ValueError
            If A is not symmetric when provided, or if neither A nor f is provided.

        Notes
        -----
        The constraint is automatically transformed to account for the Gaussian's mean and covariance.
        The transformed constraint becomes:
            y.T @ (S @ A @ S) @ y + (2 * S @ A @ mu + S @ f).T @ y + (mu.T @ A @ mu + mu.T @ f + c) >= 0
        where y = S^{-1} (x - mu) and S = Sigma_half.
        Depending on whether A and f are non-zero, the appropriate constraint type is chosen.
        

#### `sample(x0=None, n_samples=100, burn_in=100, verbose=False, cont=False)`


        Generates samples from the truncated multivariate Gaussian distribution.

        Parameters
        ----------
        x0 : Array
            Initial point for the sampler. Optional if cont is True.
        n_samples : int, optional
            Number of samples to generate, default is 100.
        burn_in : int, optional
            Number of burn-in iterations, default is 100.
        verbose : bool, optional
            Whether to print verbose output, default is False.
        cont : bool, optional
            Whether to continue from the last sampled point. Default is False.

        Returns
        -------
        Array
            Generated samples.

        Raises
        ------
        ValueError
            If cont is False and x0 is not provided, or if x0 does not satisfy constraints.
        

#### `sample_xdot()`


        Samples a new momentum vector xdot from the standard normal distribution handling GPU if necessary.
        

#### `save(filename)`


        Saves the sampler state to a pickled file.
        

---

## constraints

### Class: `BaseQuadraticConstraint`


    Base class for quadratic constraints
    

#### `A_dot_x(x)`


        Compute A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate A x at

        Returns
        -------
        Array
            Result of A x computation
        

#### `__init__(args, kwargs)`

*No documentation available.*

#### `compute_q(a, b)`


        Compute the coefficients of the constraint equation along the trajectory defined by a and b
        

#### `compute_q_(a, b)`

Placeholder method for dense q term computation

#### `compute_q_sparse(a, b)`

Placeholder method for sparse q term computation

#### `hit_time(a, b)`


        Compute the hit time of the constraint along the trajectory defined by a and b
        

#### `is_satisfied(x)`


        Check if the constraint is satisfied at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise
        

#### `is_zero(x)`


        Check if the constraint is zero at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        Tuple[bool, bool]
            (is_strictly_zero, is_approximately_zero)
        

#### `normal(x)`


        Compute the normal vector of the constraint at x
        

#### `normal_(x)`

Placeholder method for dense normal vector computation

#### `normal_sparse(x)`

Placeholder method for sparse normal vector computation

#### `reflect(x, xdot)`


        Reflect the velocity xdot at the constraint surface defined by x

        Parameters
        ----------
        x : Array
            Point on the constraint surface
        xdot : Array
            Velocity to be reflected

        Returns
        -------
        Array
            Reflected velocity
        

#### `serialize()`


        Serialize the constraint to a dictionary

        Returns
        -------
        dict
            Dictionary representation of the constraint
        

#### `value(x)`


        Compute the value of the constraint at x
        

#### `value_(x)`

Placeholder method for dense value computation

#### `value_sparse(x)`

Placeholder method for sparse value computation

#### `x_dot_A_dot_x(x)`


        Compute x^T A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate x^T A x at

        Returns
        -------
        float
            Result of x^T A x computation
        

### Class: `Constraint`


    Abstract base class for constraints
    

#### `__init__(args, kwargs)`

*No documentation available.*

#### `compute_q(a, b)`


        Compute the coefficients of the constraint equation along the trajectory defined by a and b
        

#### `hit_time(a, b)`


        Compute the hit time of the constraint along the trajectory defined by a and b
        

#### `is_satisfied(x)`


        Check if the constraint is satisfied at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise
        

#### `is_zero(x)`


        Check if the constraint is zero at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        Tuple[bool, bool]
            (is_strictly_zero, is_approximately_zero)
        

#### `normal(x)`


        Compute the normal vector of the constraint at x
        

#### `reflect(x, xdot)`


        Reflect the velocity xdot at the constraint surface defined by x

        Parameters
        ----------
        x : Array
            Point on the constraint surface
        xdot : Array
            Velocity to be reflected

        Returns
        -------
        Array
            Reflected velocity
        

#### `serialize()`


        Serialize the constraint to a dictionary

        Returns
        -------
        dict
            Dictionary representation of the constraint
        

#### `value(x)`


        Compute the value of the constraint at x
        

### Class: `LinearConstraint`


    Constraint of the form fx + c >= 0
    

#### `__init__(f, c)`

 
        Parameters
        ----------
        f : Array
            Coefficient vector
        c : float
            Constant term
        

#### `compute_q(a, b)`


        Compute the 2 q terms for the linear constraint

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float]
            q terms for the constraint

        Notes
        -----
        These expressions are defined such that Eqn 2.22 in Pakman and Paninski (2014) 
        simplifies to: q1 sin(t) + q2 cos(t) + c = 0
        

#### `hit_time(x, xdot)`


        Compute the hit time of the constraint along the trajectory defined by x and xdot

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            Hit time of the constraint along the trajectory

        Notes
        -----
        Hit time is computed by solving Eqn 2.26 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Due to the sum of inverse trig functions, we check the solution and 
        the solution +- pi to ensure we capture all hit times. 

        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.
        

#### `is_satisfied(x)`


        Check if the constraint is satisfied at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise
        

#### `is_zero(x)`


        Check if the constraint is zero at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        Tuple[bool, bool]
            (is_strictly_zero, is_approximately_zero)
        

#### `normal(x)`


        Compute the normal vector of the constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector of the constraint at x given by f
        

#### `reflect(x, xdot)`


        Reflect the velocity xdot at the constraint surface defined by x

        Parameters
        ----------
        x : Array
            Point on the constraint surface
        xdot : Array
            Velocity to be reflected

        Returns
        -------
        Array
            Reflected velocity
        

#### `serialize()`


        Serialize the constraint to a dictionary

        Returns
        -------
        dict
            Dictionary representation of the constraint
        

#### `value(x)`


        Compute the value of the constraint at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the constraint at x given by f^T x + c
        

### Class: `QuadraticConstraint`


    Constraint of the form x**T A x + b**T x + c >= 0
    

#### `A_dot_x(x)`


        Compute A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate A x at

        Returns
        -------
        Array
            Result of A x computation
        

#### `__init__(A, b, c, S, sparse=True, compiled=True)`


        Parameters
        ----------
        A : Array
            The quadratic term matrix
        b : Array
            The linear term vector
        c : float
            The constant term
        S : Array
            The transformation matrix given by the Symmetric Sqrt of the Mass matrix
        sparse : bool
            Whether to use sparse matrix computations, by default True
        compiled : bool
            Whether to use compiled code, by default True

        Notes
        -----
        If A is a sparse matrix, sparse computations are used regardless of the
        sparse parameter.
        It is highly recommended to use compiled code for performance reasons.
        

#### `compute_q(a, b)`


        Compute the coefficients of the constraint equation along the trajectory defined by a and b
        

#### `compute_q_(a, b)`


        Compute the 5 q terms for the quadratic constraint using dense matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float, float, float]
            q terms for the quadratic constraint

        Notes
        -----
        These expressions are defined in Eqns 2.40-2.44 in Pakman and Paninski (2014)
        

#### `compute_q_sparse(a, b)`


        Compute the 5 q terms for the quadratic constraint using sparse matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float, float, float]
            q terms for the quadratic constraint

        Notes
        -----
        These expressions are defined in Eqns 2.40-2.44 in Pakman and Paninski (2014)
        

#### `hit_time(a, b)`


        Compute the hit time of the constraint along the trajectory defined by a and b
        

#### `hit_time_cpp(x, xdot)`


        Compute the hit time for the quadratic constraint using compiled code

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            The hit time for the constraint

        Notes
        -----
        Hit time is computed by solving Eqn 2.48 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.

        Compiled code is both written in C++ and optimized to remove all redundant computations
        see paper for details.
        

#### `hit_time_py(x, xdot)`


        Compute the hit time for the quadratic constraint using Python code

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            The hit time for the constraint

        Notes
        -----
        Hit time is computed by solving Eqn 2.48 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.

        It is highly recommended to use the compiled version for performance reasons. 
        This Python version is maintained for testing and validation purposes.
        

#### `is_satisfied(x)`


        Check if the constraint is satisfied at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise
        

#### `is_zero(x)`


        Check if the constraint is zero at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        Tuple[bool, bool]
            (is_strictly_zero, is_approximately_zero)
        

#### `normal(x)`


        Compute the normal vector of the constraint at x
        

#### `normal_(x)`


        Compute the normal vector at x using dense matrix computations
        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x + b
        

#### `normal_sparse(x)`


        Compute the normal vector at x using sparse matrix computations
        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x + b
        

#### `reflect(x, xdot)`


        Reflect the velocity xdot at the constraint surface defined by x

        Parameters
        ----------
        x : Array
            Point on the constraint surface
        xdot : Array
            Velocity to be reflected

        Returns
        -------
        Array
            Reflected velocity
        

#### `serialize()`


        Serialize the constraint to a dictionary

        Returns
        -------
        dict
            Dictionary representation of the constraint
        

#### `value(x)`


        Compute the value of the constraint at x
        

#### `value_(x)`


        Compute the value of the constraint at x using dense matrix computations
        
        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        float
            The value of the constraint at x given by x^T A x + b^T x + c
        

#### `value_sparse(x)`


        Compute the value of the constraint at x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            The value of the constraint at x given by x^T A x + b^T x + c
        

#### `x_dot_A_dot_x(x)`


        Compute x^T A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate x^T A x at

        Returns
        -------
        float
            Result of x^T A x computation
        

### Class: `SimpleQuadraticConstraint`


    Constraint of the form x^T A x + c >= 0
    

#### `A_dot_x(x)`


        Compute A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate A x at

        Returns
        -------
        Array
            Result of A x computation
        

#### `__init__(A, c, S, sparse=False)`


        Parameters
        ----------
        A : Array
            Quadratic coefficient matrix
        c : float
            Constant term
        S : Array
            Transformation matrix given by the Symmetric Sqrt of the Mass matrix
        sparse : bool, optional
            Whether to use sparse matrix computations, by default False

        Notes
        -----
        If A is a sparse matrix, sparse computations are used regardless of the
        sparse parameter.
        

#### `compute_q(a, b)`


        Compute the coefficients of the constraint equation along the trajectory defined by a and b
        

#### `compute_q_(a, b)`


        Compute the 3 q terms for the simple quadratic constraint using dense matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float]
            q terms for the constraint

        Notes
        -----
        These expressions are the nonzero q terms defined in equation 2.45 in Pakman and Paninski (2014)
        

#### `compute_q_sparse(a, b)`


        Compute the 3 q terms for the simple quadratic constraint using sparse matrix computations

        Parameters
        ----------
        a : Array
            The velocity of the point in the HMC trajectory
        b : Array
            The position of the point in the HMC trajectory

        Returns
        -------
        Tuple[float, float, float]
            q terms for the constraint

        Notes
        -----
        These expressions are the nonzero q terms defined in equation 2.45 in Pakman and Paninski (2014)
        

#### `hit_time(x, xdot)`


        Compute the hit time for the simple quadratic constraint

        Parameters
        ----------
        x : Array
            The position of the point in the HMC trajectory
        xdot : Array
            The velocity of the point in the HMC trajectory

        Returns
        -------
        Array
            The hit time for the constraint

        Notes
        -----
        Hit time is computed by solving Eqn 2.45 in Pakman and Paninski (2014)
        See resources/HMC_exact_soln.nb for derivation
        Only positive hit times are returned and any ghost solutions are filtered 
        out at a later stage.
        

#### `is_satisfied(x)`


        Check if the constraint is satisfied at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise
        

#### `is_zero(x)`


        Check if the constraint is zero at x

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at
        Returns
        -------
        Tuple[bool, bool]
            (is_strictly_zero, is_approximately_zero)
        

#### `normal(x)`


        Compute the normal vector of the constraint at x
        

#### `normal_(x)`


        Compute the normal vector at x using dense matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x
        

#### `normal_sparse(x)`


        Compute the normal vector at x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the normal vector at

        Returns
        -------
        Array
            Normal vector at x given by 2 * A @ x
        

#### `reflect(x, xdot)`


        Reflect the velocity xdot at the constraint surface defined by x

        Parameters
        ----------
        x : Array
            Point on the constraint surface
        xdot : Array
            Velocity to be reflected

        Returns
        -------
        Array
            Reflected velocity
        

#### `serialize()`


        Serialize the constraint to a dictionary

        Returns
        -------
        dict
            Dictionary representation of the constraint
        

#### `value(x)`


        Compute the value of the constraint at x
        

#### `value_(x)`


        Compute the value of the constraint at x using dense matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the constraint at x given by x^T A x + c
        

#### `value_sparse(x)`


        Compute the value of the constraint at x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate the constraint at

        Returns
        -------
        float
            Value of the constraint at x given by x^T A x + c
        

#### `x_dot_A_dot_x(x)`


        Compute x^T A x using sparse matrix computations

        Parameters
        ----------
        x : Array
            Point to evaluate x^T A x at

        Returns
        -------
        float
            Result of x^T A x computation
        

---

## utils

### `arccos(x: float)`


    Computes the real component of the arccosine of a value.

    Parameters
    ----------
    x : float
        The input value.

    Returns
    -------
    float
        The real component of the arccosine of the input value.

    Notes
    -----
    Uses the cmath.acos function to handle complex values and returns the real part.
    Can potentially create ghost values if the input is outside the range [-1, 1]. 
    However, due to the complexity of the solution expressions this is necessary for 
    numerical stability and ghost solutions are filtered out later.
    

### `get_shared_library()`


    Loads the compiled shared library for calculating the quadratic constraint hit times.

    Returns
    -------
        ctypes.CDLL: The loaded shared library.

    Notes
    -----
    The shared library is expected to be located at 'compiled/calc_solutions.so'
    relative to the base path of this module.

    Shared Library Function:
    - calc_all_solutions: Calculates all solutions for the quadratic constraint hit times.
        - Arguments: Five double precision floating-point numbers.
        - Returns: A pointer to an array of double precision floating-point numbers.
    

### `get_sparse_elements(A)`


    Extracts the row, column, and data elements from a sparse matrix.

    Parameters
    ----------
    A : Array
        The input sparse matrix.

    Returns
    -------
    Tuple[Array, Array, Array]
        A tuple containing the row indices, column indices, and data values of the sparse matrix.
    

### `sparsify(A)`


    Converts a dense numpy array or a PyTorch tensor to a sparse COO matrix.

    Parameters
    ----------
    A : Array
        The input array to be converted to a sparse matrix.

    Returns
    -------
    Array
        The sparse COO matrix representation of the input array.
    

### `to_scalar(x)`


    Converts a scalar array or a float to a float.

    Parameters
    ----------
    x : Array | float
        The input value to be converted.

    Returns
    -------
    float
        The converted float value.
    

---

## quadratic_solns

### `soln1(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the first of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

### `soln2(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the second of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

### `soln3(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the third of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

### `soln4(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the fourth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

### `soln5(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the fifth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

### `soln6(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the sixth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

### `soln7(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the seventh of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

### `soln8(q1: float, q2: float, q3: float, q4: float, q5: float)`


    Computes the eighth of the 8 solutions for the full quadratic constraint hit time.

    Parameters
    ----------
    q1 : float
        The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
    q2 : float
        The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
    q3 : float
        The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
    q4 : float
        The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
    q5 : float
        The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).

    Returns
    -------
    float
        The computed solution.

    Notes
    -----
    DO NOT MODIFY THIS FUNCTION
    The solution is derived from the quartic equation associated with the quadratic constraint hit time
    given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using 
    Mathematica and then exported to Fortran which uses the same syntax as Python for mathematical operations.
    See resources/HMC_exact_soln.nb for details.

    It is not recommended to use this function directly due to its complexity and slow performance. Instead,
    use the compiled shared library accessed via `get_shared_library()` for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    

---

