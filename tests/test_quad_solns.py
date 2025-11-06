import pytest 
import pytest_cov 
import numpy as np
from hypothesis import given, assume, settings, HealthCheck
import hypothesis.strategies as st
from tmg_hmc.compiled import (calc_all_solutions, soln1 as c_soln1, 
                              soln2 as c_soln2, soln3 as c_soln3, 
                              soln4 as c_soln4, soln5 as c_soln5, 
                              soln6 as c_soln6, soln7 as c_soln7, 
                              soln8 as c_soln8)
from tmg_hmc.quad_solns import (soln1, soln2, soln3, soln4, 
                                soln5, soln6, soln7, soln8)

eps = 1e-3
btol = 5e-2

def close_arccos(x, y, tol=eps) -> bool:
    # Check arccos boundary cases
    if (np.isnan(x) and np.isclose(np.abs(y), np.pi, atol=btol)) or (np.isnan(y) and np.isclose(np.abs(x), np.pi, atol=btol)):
        return True
    elif (np.isnan(x) and np.isclose(np.abs(y), 0.0, atol=btol)) or (np.isnan(y) and np.isclose(np.abs(x), 0.0, atol=btol)):
        return True
    elif np.isclose(np.abs(x), np.pi, atol=10*btol) and np.isclose(np.abs(y), np.pi, atol=10*btol):
        return np.isclose(x, y, atol=btol)
    elif np.isclose(np.abs(x), 0.0, atol=10*btol) and np.isclose(np.abs(y), 0.0, atol=10*btol):
        return np.isclose(x, y, atol=tol*100)
    return np.isclose(x, y, equal_nan=True, atol=tol)

def no_div_by_zero(q1, q2, q3, q4, q5, eps=1e-6) -> bool:
    def pow(x, y):
        return x ** y
    
    # Common denominators
    q1_sq = q1 * q1
    q4_sq = q4 * q4
    denom = q1_sq + q4_sq

    # First sqrt term
    term1_a = pow(q1*q2 + q4*q5, 2) / (denom * denom)
    term2_a = 2.0 * (pow(q2,2) + 2*q1*q3 - q4_sq + pow(q5,2)) / (3.0 * denom)

    # Cubic term
    A = q1*q2 + q4*q5
    B = q2*q3 - q4*q5
    C = pow(q2,2) + 2*q1*q3 - q4_sq + pow(q5,2)
    C_sq = C * C
    C_cu = C_sq * C
    B_sq = B * B
    A_sq = A * A

    cubic_num = -12.0*B*A + 12.0*denom*(pow(q3,2) - pow(q5,2)) + C_sq

    big_term = 108.0*denom*B_sq + \
        108.0*A_sq*(pow(q3,2) - pow(q5,2)) - \
        36.0*B*A*C - 72.0*denom*(pow(q3,2) - \
        pow(q5,2))*C + 2.0*C_cu

    inner_sqrt = np.sqrt(-4.0*pow(cubic_num,3) + pow(big_term,2))
    cubic_root = pow(big_term + inner_sqrt, 1.0/3.0)

    pow2_1_3 = pow(2.0, 1.0/3.0)
    term3_a = pow2_1_3 * cubic_num / (3.0 * denom * cubic_root)
    term4_a = cubic_root / (3.0 * pow2_1_3 * denom)

    first_sqrt = np.sqrt(term1_a - term2_a + term3_a + term4_a)

    return (denom**3 > eps) and (cubic_root > eps) and (first_sqrt > eps)

skip_zero = st.floats(-1e5, -1e-6) | st.floats(1e-6, 1e5)
def given_qs(func):
    # Decorator to apply the same strategy to all five qs
    # Zero values cause divisions by zero and numerical instability
    return given(q1=skip_zero, q2=skip_zero, q3=skip_zero, q4=skip_zero, q5=skip_zero)(func)

@given_qs
def test_soln1_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s1 = soln1(q1, q2, q3, q4, q5)
        c_s1 = c_soln1(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s1, c_s1), f"soln1 mismatch: {s1} vs {c_s1}"

@given_qs
def test_soln2_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s2 = soln2(q1, q2, q3, q4, q5)
        c_s2 = c_soln2(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s2, c_s2), f"soln2 mismatch: {s2} vs {c_s2}"

@given_qs
def test_soln3_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s3 = soln3(q1, q2, q3, q4, q5)
        c_s3 = c_soln3(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s3, c_s3), f"soln3 mismatch: {s3} vs {c_s3}"

@given_qs
def test_soln4_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s4 = soln4(q1, q2, q3, q4, q5)
        c_s4 = c_soln4(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s4, c_s4), f"soln4 mismatch: {s4} vs {c_s4}"

@given_qs
def test_soln5_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s5 = soln5(q1, q2, q3, q4, q5)
        c_s5 = c_soln5(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s5, c_s5), f"soln5 mismatch: {s5} vs {c_s5}"

@given_qs
def test_soln6_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s6 = soln6(q1, q2, q3, q4, q5)
        c_s6 = c_soln6(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s6, c_s6), f"soln6 mismatch: {s6} vs {c_s6}"

@given_qs
def test_soln7_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s7 = soln7(q1, q2, q3, q4, q5)
        c_s7 = c_soln7(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s7, c_s7), f"soln7 mismatch: {s7} vs {c_s7}"

@given_qs
def test_soln8_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    zero_div = False
    try:
        s8 = soln8(q1, q2, q3, q4, q5)
        c_s8 = c_soln8(q1, q2, q3, q4, q5)
    except ZeroDivisionError:
        zero_div = True
    assume(not zero_div)
    assert close_arccos(s8, c_s8), f"soln8 mismatch: {s8} vs {c_s8}"

@given_qs
def test_calc_all_consistency(q1, q2, q3, q4, q5):
    assume(no_div_by_zero(q1, q2, q3, q4, q5, eps))  # Division by zero otherwise
    sols = calc_all_solutions(q1, q2, q3, q4, q5)
    s1 = c_soln1(q1, q2, q3, q4, q5)
    s2 = c_soln2(q1, q2, q3, q4, q5)
    s3 = c_soln3(q1, q2, q3, q4, q5)
    s4 = c_soln4(q1, q2, q3, q4, q5)
    s5 = c_soln5(q1, q2, q3, q4, q5)
    s6 = c_soln6(q1, q2, q3, q4, q5)
    s7 = c_soln7(q1, q2, q3, q4, q5)
    s8 = c_soln8(q1, q2, q3, q4, q5)
    assert close_arccos(sols[0], s1), f"soln1 mismatch: {sols[0]} vs {s1}"
    assert close_arccos(sols[1], s2), f"soln2 mismatch: {sols[1]} vs {s2}"
    assert close_arccos(sols[2], s3), f"soln3 mismatch: {sols[2]} vs {s3}"
    assert close_arccos(sols[3], s4), f"soln4 mismatch: {sols[3]} vs {s4}"
    assert close_arccos(sols[4], s5), f"soln5 mismatch: {sols[4]} vs {s5}"
    assert close_arccos(sols[5], s6), f"soln6 mismatch: {sols[5]} vs {s6}"
    assert close_arccos(sols[6], s7), f"soln7 mismatch: {sols[6]} vs {s7}"
    assert close_arccos(sols[7], s8), f"soln8 mismatch: {sols[7]} vs {s8}"
