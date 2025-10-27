#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>

using namespace std;

double arccos(complex<double> z){
    return real(acos(z));
}

extern "C" {
    /**
     * Computes all 8 solutions for the full quadratic constraint hit time.
     * 
     * This function is exposed via C linkage for use with Python ctypes and provides
     * efficient computation of all solutions simultaneously.
     * 
     * @param q1 The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
     * @param q2 The second parameter defined in Eqn 2.41 of Pakman and Paninski (2014).
     * @param q3 The third parameter defined in Eqn 2.42 of Pakman and Paninski (2014).
     * @param q4 The fourth parameter defined in Eqn 2.43 of Pakman and Paninski (2014).
     * @param q5 The fifth parameter defined in Eqn 2.44 of Pakman and Paninski (2014).
     * 
     * @return Pointer to dynamically allocated array of 8 doubles containing all computed solutions.
     *         The caller must free this memory using free_ptr() to avoid memory leaks.
     * 
     * @note DO NOT MODIFY THE IMPLEMENTATION OF THIS FUNCTION
     * @note The solutions are derived from the quartic equation associated with the quadratic 
     *       constraint hit time given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). 
     *       The expressions are computed exactly using Mathematica and then exported to C.
     *       Then the resulting C++ code is optimized for performance by removing all 
     *       redundant calculations. See paper for details on how this is done.
     * 
     * @warning The returned pointer must be freed using free_ptr() to prevent memory leaks.
     * 
     * @see free_ptr()
     */
    double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5);
    
    /**
     * Frees memory allocated by calc_all_solutions.
     * 
     * @param arr Pointer to array returned by calc_all_solutions().
     */
    void free_ptr(double* arr){
        delete[] arr;
    }
}