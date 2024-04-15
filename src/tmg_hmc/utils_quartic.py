import numpy as np 
# ignore runtime warning
np.seterr(divide='ignore', invalid='ignore')

def nanmin(arr) -> float:
    if len(arr) == 0:
        return np.nan
    elif np.isnan(arr).all():
        return np.nan
    else:
        return np.nanmin(arr)

def nanargmin(arr) -> int | None:
    if len(arr) == 0 or np.isnan(arr).all():
        return None
    else:
        return np.nanargmin(arr)

def soln1(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    return -0.25*(2*q1*q2 + 2*q4 + q5)/(q1**2 + q4**2) - \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/ \
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/ \
            (3.*2**(1/3.)*(q1**2 + q4**2)))/2. - \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(2.*(q1**2 + q4**2)**2) - \
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/ \
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) - \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/ \
            (3.*2**(1/3.)*(q1**2 + q4**2)) - \
            (-((2*q1*q2 + 2*q4 + q5)**3/(q1**2 + q4**2)**3) + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + \
            (4*(2*q1*q2 + 2*q4 + q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/ \
            (4.*np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/ \
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/ \
            (3.*2**(1/3.)*(q1**2 + q4**2)))))/2.

def soln2(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    return -0.25*(2*q1*q2 + 2*q4 + q5)/(q1**2 + q4**2) - \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)))/2. + \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(2.*(q1**2 + q4**2)**2) - \
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) - \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)) - \
            (-((2*q1*q2 + 2*q4 + q5)**3/(q1**2 + q4**2)**3) + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + \
            (4*(2*q1*q2 + 2*q4 + q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/\
            (4.*np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - 
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)))))/2.

def soln3(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    return -0.25*(2*q1*q2 + 2*q4 + q5)/(q1**2 + q4**2) + \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)))/2. - \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(2.*(q1**2 + q4**2)**2) - \
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) - \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)) + \
            (-((2*q1*q2 + 2*q4 + q5)**3/(q1**2 + q4**2)**3) + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + \
            (4*(2*q1*q2 + 2*q4 + q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/\
            (4.*np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)))))/2.

def soln4(q1: float, q2: float, q3: float, q4: float, q5: float) -> float:
    return -0.25*(2*q1*q2 + 2*q4 + q5)/(q1**2 + q4**2) + \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)))/2. + \
            np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(2.*(q1**2 + q4**2)**2) - \
            (4*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) - \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) - \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + 12*(q1**2 + q4**2)*(q3**2 - q5**2) + \
            (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)) + \
            (-((2*q1*q2 + 2*q4 + q5)**3/(q1**2 + q4**2)**3) + (16*(-(q2*q3) + q4*q5))/(q1**2 + q4**2) + \
            (4*(2*q1*q2 + 2*q4 + q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(q1**2 + q4**2)**2)/\
            (4.*np.sqrt((2*q1*q2 + 2*q4 + q5)**2/(4.*(q1**2 + q4**2)**2) - \
            (2*(q2**2 + 2*q1*q3 - q4**2 + q5**2))/(3.*(q1**2 + q4**2)) + \
            (2**(1/3.)*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2))/\
            (3.*(q1**2 + q4**2)*(108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)) + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + 27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3 + \
            np.sqrt(-4*(-6*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5) + \
            12*(q1**2 + q4**2)*(q3**2 - q5**2) + (q2**2 + 2*q1*q3 - q4**2 + q5**2)**2)**3 + \
            (108*(q1**2 + q4**2)*(q2*q3 - q4*q5)**2 + \
            27*(2*q1*q2 + 2*q4 + q5)**2*(q3**2 - q5**2) - \
            18*(2*q1*q2 + 2*q4 + q5)*(q2*q3 - q4*q5)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) - \
            72*(q1**2 + q4**2)*(q3**2 - q5**2)*(q2**2 + 2*q1*q3 - q4**2 + q5**2) + \
            2*(q2**2 + 2*q1*q3 - q4**2 + q5**2)**3)**2))**(1/3.)/\
            (3.*2**(1/3.)*(q1**2 + q4**2)))))/2.