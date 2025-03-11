from __future__ import annotations
import numpy as np
from typing import Protocol, Tuple
import torch
from tmg_hmc.utils import (Array, to_scalar, get_sparse_elements, get_shared_library,
                            soln1, soln2, soln3, soln4, soln5, soln6, soln7, soln8)

pis = np.array([-1, 0, 1]) * np.pi
eps = 1e-12

# Load the shared library
lib = get_shared_library()

class Constraint(Protocol):
    def value(self, x: Array) -> float:...
        
    def is_satisfied(self, x: Array) -> bool:
        return self.value(x) >= 0 

    def is_zero(self, x: Array) -> Tuple[bool, bool]:
        val = self.value(x)
        return np.isclose(val, 0), np.isclose(val, 0, atol=1e-2)
    
    def compute_q(self, a: Array, b: Array) -> Tuple[float, ...]:...

    def hit_time(self, a: Array, b: Array) -> Array:...

    def normal(self, x: Array) -> Array:...

    def reflect(self, x: Array, xdot: Array) -> Array:
        f = self.normal(x)
        if isinstance(xdot, torch.Tensor):
            norm = torch.sqrt(f.T @ f)
        else:
            norm = np.sqrt(f.T @ f)
        f = f / norm
        return xdot - 2 * (f.T @ xdot) * f

    def serialize(self) -> dict:
        d = self.__dict__.copy()
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.cpu()
        d['type'] = self.__class__.__name__
        return d
    
    @classmethod
    def deserialize(cls, d: dict, gpu: bool) -> Constraint:
        if gpu:
            for k, v in d.items():
                if isinstance(v, torch.Tensor):
                    d[k] = v.cuda()
        if d['type'] == 'LinearConstraint':
            return LinearConstraint(d['f'], d['c'])
        elif d['type'] == 'SimpleQuadraticConstraint':
            return SimpleQuadraticConstraint.build_from_dict(d, gpu)
        elif d['type'] == 'QuadraticConstraint':
            return QuadraticConstraint.build_from_dict(d, gpu)
        else:
            raise ValueError(f"Unknown constraint type {d['type']}")
    
    

class LinearConstraint(Constraint):
    """
    Constraint of the form fx + c >= 0
    """
    def __init__(self, f: Array, c: float):
        self.f = f
        self.c = c
    
    def value(self, x: Array) -> float:
        return to_scalar(self.f.T @ x + self.c)

    def normal(self, x: Array) -> Array:
        return self.f

    def compute_q(self, a: Array, b: Array) -> Tuple[float, float]:
        f = self.f
        q1 = to_scalar(f.T @ a)
        q2 = to_scalar(f.T @ b)
        return q1, q2

    def hit_time(self, x: Array, xdot: Array) -> Array:
        a, b = xdot, x
        q1, q2 = self.compute_q(a, b)
        c = self.c
        u = np.sqrt(q1**2 + q2**2)
        if (u < abs(c)) or (u == 0) or (q2 == 0): # No intersection
            return np.array([np.nan])
        s1 = -np.arccos(-c/u) + np.arctan(q1/q2) + pis
        s2 = np.arccos(-c/u) + np.arctan(q1/q2) + pis
        s = np.hstack([s1, s2])
        return s[s > eps]



class BaseQuadraticConstraint(Constraint):
    """
    Base class for quadratic constraints
    """
    def _setup_values(self, A: Array, S: Array):
        self.A_orig = A
        self.S = S
        self.value = self.value_
        self.normal = self.normal_
        self.compute_q = self.compute_q_

    def _setup_values_sparse(self, A: Array, S: Array):
        rows, cols, vals = get_sparse_elements(A)
        self.n_comps = len(rows)
        self.n = A.shape[0]
        self.A_orig = A
        self.s_rows = np.vstack([S[i,:].reshape((1,self.n)) for i in rows]) # S[i,:] is a row vector
        self.s_cols = np.hstack([S[:,j].reshape((self.n,1)) for j in cols]) # S[:,j] is a column vector
        self.a_vals = vals
        self.value = self.value_sparse
        self.normal = self.normal_sparse
        self.compute_q = self.compute_q_sparse

    def value_(self, x: Array) -> float:...

    def value_sparse(self, x: Array) -> float:...

    def normal_(self, x: Array) -> Array:...

    def normal_sparse(self, x: Array) -> Array:...

    def compute_q_(self, a: Array, b: Array) -> Tuple[float, ...]:...

    def compute_q_sparse(self, a: Array, b: Array) -> Tuple[float, ...]:...

    @property 
    def A(self):
        return self.S @ self.A_orig @ self.S
    
    def A_dot_x_py(self, x: Array) -> Array:
        dot_prods = [self.s_rows[i,:].reshape((1,self.n)) @ x for i in range(self.n_comps)]#[row @ x for row in self.s_rows]
        return sum([self.a_vals[i]*dot_prods[i]*self.s_cols[:,i].reshape((self.n,1)) for i in range(self.n_comps)])#sum([val * dot * col for val, dot, col in zip(self.a_vals, self.s_cols, dot_prods)])
    
    def A_dot_x_cpp(self, x: Array) -> Array:
        return lib.dot_sparse(x, self.s_rows, self.a_vals, self.n, self.n_comps)

    def x_dot_A_dot_x_py(self, x: Array) -> float:
        return x.T @ self.A_dot_x(x)
    
    def x_dot_A_dot_x_cpp(self, x: Array) -> float:
        return lib.dot_sparse_dot(x, self.s_rows, self.a_vals, self.n, self.n_comps)



class SimpleQuadraticConstraint(BaseQuadraticConstraint):
    """
    Constraint of the form x^T A x + c >= 0
    """
    def __init__(self, A: Array, c: float, S: Array, sparse: bool = False):
        # Check that A is symmetric
        self.c = c
        self.sparse = sparse
        if sparse:
            self._setup_values_sparse(A, S)
        else:
            self._setup_values(A, S)

    @classmethod 
    def build_from_dict(cls, d: dict, gpu: bool) -> SimpleQuadraticConstraint:
        sparse = d['sparse']
        A = d['A_orig']
        c = d['c']
        if sparse:
            cols = d['s_cols']
            rows = d['s_rows']
            n = len(cols[0])
            S = torch.zeros((n,n)).cuda() if gpu else np.zeros((n,n))
            rowinds, colinds, _ = get_sparse_elements(A)
            for rowind in rowinds:
                S[rowind,:] = rows[rowind]
            for colind in colinds:
                S[:,colind] = cols[colind]
            return cls(A, c, S, sparse)
        else:
            S = d['S']
            return cls(A, c, S, sparse)
    
    def value_(self, x: Array) -> float:
        return to_scalar(x.T @ self.A @ x + self.c)
    
    def value_sparse(self, x: Array) -> float:
        return to_scalar(self.x_dot_A_dot_x(x) + self.c)

    def normal_(self, x: Array) -> Array:
        return 2 * self.A @ x
    
    def normal_sparse(self, x: Array) -> Array:
        return 2 * self.A_dot_x(x)

    def compute_q_(self, a: Array, b: Array) -> Tuple[float, float, float]:
        A = self.A
        c = self.c
        q1 = to_scalar(b.T @ A @ b - a.T @ A @ a)
        q3 = c + to_scalar(a.T @ A @ a)
        q4 = to_scalar(2 * a.T @ A @ b)
        return q1, q3, q4
    
    def compute_q_sparse(self, a: Array, b: Array) -> Tuple[float, float, float]:
        q1 = to_scalar(self.x_dot_A_dot_x(b) - self.x_dot_A_dot_x(a))
        q3 = self.c + to_scalar(self.x_dot_A_dot_x(a))
        q4 = to_scalar(2 * a.T @ self.A_dot_x(b))
        return q1, q3, q4
    
    def hit_time(self, x: Array, xdot: Array) -> Array:
        a, b = xdot, x
        q1, q3, q4 = self.compute_q(a, b)
        u = np.sqrt(q1**2 + q4**2)
        if (u == 0) or (q4 == 0): # No intersection
            return np.array([np.nan])
        s1 = (np.pi + np.arcsin((q1+2*q3)/u) -
              np.arctan(q1/q4) + pis) / 2 
        s2 = (-np.arcsin((q1+2*q3)/u) -
              np.arctan(q1/q4)+ pis) / 2 
        s = np.hstack([s1, s2])
        return s[s > eps]



class QuadraticConstraint(BaseQuadraticConstraint):
    """
    Constraint of the form x**T A x + b**T x + c >= 0
    """
    def __init__(self, A: Array, b: Array, c: float, S: Array, sparse: bool = True, compiled: bool = True):
        self.c = c
        self.b = b
        self.sparse = sparse or compiled
        if self.sparse:
            self._setup_values_sparse(A, S)
        else:
            self._setup_values(A, S)
        if self.compiled:
            self.hit_time = self.hit_time_cpp
            self.s_rows = self.s_rows.flatten().astype(np.float64)
            self.A_dot_x = self.A_dot_x_cpp
            self.x_dot_A_dot_x = self.x_dot_A_dot_x_cpp
        else:
            self.hit_time = self.hit_time_py
            self.A_dot_x = self.A_dot_x_py
            self.x_dot_A_dot_x = self.x_dot_A_dot_x_py

    @classmethod 
    def build_from_dict(cls, d: dict, gpu: bool) -> SimpleQuadraticConstraint:
        sparse = d['sparse']
        A = d['A_orig']
        c = d['c']
        b = d['b']
        if sparse:
            cols = d['s_cols']
            rows = d['s_rows']
            print(rows)
            n = len(cols[0].flatten())
            dtype = cols[0].dtype
            print(n)
            S = torch.zeros((n,n), dtype=dtype).cuda() if gpu else np.zeros((n,n), dtype=dtype)
            rowinds, colinds, _ = get_sparse_elements(A)
            for i in range(len(rowinds)):
                rowind = rowinds[i]
                S[rowind,:] = rows[i].flatten()
            for j in range(len(colinds)):
                colind = colinds[j]
                S[:,colind] = cols[j].flatten()
            return cls(A, b, c, S, sparse)
        else:
            S = d['S']
            return cls(A, b, c, S, sparse)
    
    def value_(self, x: Array) -> float:
        return to_scalar(x.T @ self.A @ x + self.b.T @ x + self.c)
    
    def value_sparse(self, x: Array) -> float:
        return to_scalar(self.x_dot_A_dot_x(x) + self.b.T @ x + self.c)

    def normal_(self, x: Array) -> Array:
        return 2 * self.A @ x + self.b
    
    def normal_sparse(self, x: Array) -> Array:
        return 2 * self.A_dot_x(x) + self.b

    def compute_q_(self, a: Array, b: Array) -> Tuple[float, float, float, float, float]:
        A = self.A
        B = self.b
        c = self.c
        q1 = to_scalar(b.T @ A @ b - a.T @ A @ a)
        q2 = to_scalar(B.T @ b)
        q3 = c + to_scalar(a.T @ A @ a)
        q4 = to_scalar(2 * a.T @ A @ b)
        q5 = to_scalar(B.T @ a)
        return q1, q2, q3, q4, q5
    
    def compute_q_sparse(self, a: Array, b: Array) -> Tuple[float, float, float, float, float]:
        B = self.b
        c = self.c
        q1 = to_scalar(self.x_dot_A_dot_x(b) - self.x_dot_A_dot_x(a))
        q2 = to_scalar(B.T @ b)
        q3 = c + to_scalar(self.x_dot_A_dot_x(a))
        q4 = to_scalar(2 * a.T @ self.A_dot_x(b))
        q5 = to_scalar(B.T @ a)
        return q1, q2, q3, q4, q5

    def hit_time_cpp(self, x: Array, xdot: Array) -> Array:
        a, b = xdot, x
        pis = np.array([-2, 0, 2]).reshape(3,1)*np.pi
        qs = self.compute_q(a, b)
        soln = lib.calc_all_solutions(*qs)
        s = np.ctypeslib.as_array(soln, shape=(1,8))
        lib.free_ptr(soln)
        s = (s + pis).flatten()
        return np.unique(s[s > eps])#1e-8])
    
    def hit_time_py(self, x: Array, xdot: Array) -> Array:
        a, b = xdot, x
        pis = np.array([-2, 0, 2]).reshape(3,1)*np.pi
        qs = self.compute_q(a, b)
        s1 = soln1(*qs) + pis
        s2 = soln2(*qs) + pis
        s3 = soln3(*qs) + pis
        s4 = soln4(*qs) + pis
        s5 = soln5(*qs) + pis
        s6 = soln6(*qs) + pis
        s7 = soln7(*qs) + pis
        s8 = soln8(*qs) + pis
        s = np.hstack([s1, s2, s3, s4, s5, s6, s7, s8])
        return np.unique(s[s > eps])#1e-8])
    
if __name__ == "__main__":
    test_constr = QuadraticConstraint(np.array([[1, 0], [0, 1]]), np.array([0, 0]), 0, np.array([[1, 0], [0, 1]]))
    x = np.array([1, 1]).reshape(2,1)
    xdot = np.array([1, 0]).reshape(2,1)
    import time 
    start = time.time()
    for _ in range(100000):
        test_constr.hit_time_py(x, xdot)
    py_time = time.time() - start
    print(f'Python time: {py_time}')
    start = time.time()
    for _ in range(100000):
        test_constr.hit_time_cpp(x, xdot)
    cpp_time = time.time() - start
    print(f'C++ time: {cpp_time}')
    print(f'Speedup: {py_time / cpp_time}')
