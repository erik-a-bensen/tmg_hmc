from cmath import sqrt
from tmg_hmc.utils import arccos 

def soln1(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) - 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln2(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) - 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln3(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) - 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln4(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) - 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) - 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln5(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) + 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln6(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. - 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) + 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln7(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (-arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
         sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
         sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
            (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
             (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                  108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                  36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                  72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                  2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                  sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                        (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                    (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                      (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                  (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
             (3.*2**(1./3)*(q1**2 + q4**2)) + 
            ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
               (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
             (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                 (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                 (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                  (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                       108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                       36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                       72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                       2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                       sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                             12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3
                           + (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                            108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                     72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                     2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                     sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                           (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                       (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                          72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                          2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                  (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))

def soln8(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    """
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
    use the compiled module for efficient computation of all solutions.
    This function is maintained for reference and validation purposes.
    """
    return (arccos(-0.5*(q1*q2 + q4*q5)/(q1**2 + q4**2) + 
        sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
           (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)))/2. + 
        sqrt((2*(q1*q2 + q4*q5)**2)/(q1**2 + q4**2)**2 - 
           (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - 
           (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                 72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                 2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                 sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                       (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                   (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) - 
           (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
               72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
               2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
               sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                     (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                 (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
            (3.*2**(1./3)*(q1**2 + q4**2)) + 
           ((-8*(q1*q2 + q4*q5)**3)/(q1**2 + q4**2)**3 + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + 
              (8*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/
            (4.*sqrt((q1*q2 + q4*q5)**2/(q1**2 + q4**2)**2 - 
                (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + 
                (2**(1./3)*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                     12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/
                 (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                      108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                      72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                      2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                      sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                        (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 
                           108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                           36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                           72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                           2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)) + 
                (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                    36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                    72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                    2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + 
                    sqrt(-4*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + 
                          (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + 
                      (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 108*(q1*q2 + q4*q5)**2*(q3**2 - q5**2) - 
                         36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
                         72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + 
                         2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1./3)/
                 (3.*2**(1./3)*(q1**2 + q4**2)))))/2.))
