#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>
#include <vector>

namespace py = pybind11;

double arccos(std::complex<double> z) {
    return std::real(std::acos(z));
}

double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5);

/**
 * Pybind11 module for computing all 8 solutions for the full quadratic constraint hit time.
 */
PYBIND11_MODULE(compiled, m) {
    m.doc() = R"pbdoc(
        Pybind11 wrapper for the quadratic constraint hit time solver.
    )pbdoc";

    m.def(
        "calc_all_solutions",
        [](double q1, double q2, double q3, double q4, double q5) {
            double* result = calc_all_solutions(q1, q2, q3, q4, q5);
            if (!result)
                throw std::runtime_error("calc_all_solutions returned a null pointer");

            // Copy results into NumPy array
            py::array_t<double> out(8);
            auto buf = out.mutable_unchecked<1>();
            for (int i = 0; i < 8; ++i)
                buf(i) = result[i];

            delete[] result;
            return out;
        },
        R"pbdoc(
            calc_all_solutions(x: np.ndarray, xdot: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray

            Compute all 8 solutions for the full quadratic constraint hit time.

            This function computes all eight possible hit times for the quadratic constraint
            used in Hamiltonian Monte Carlo sampling, following the derivations in Pakman
            and Paninski (2014).

            Parameters
            ----------
            q1 : float
                The first parameter defined in Eqn 2.40 of Pakman and Paninski (2014).
            q2 : float
                The second parameter defined in Eqn 2.41.
            q3 : float
                The third parameter defined in Eqn 2.42.
            q4 : float
                The fourth parameter defined in Eqn 2.43.
            q5 : float
                The fifth parameter defined in Eqn 2.44.

            Returns
            -------
            numpy.ndarray
                A 1D NumPy array of length 8 containing all computed solutions.

            Notes
            -----
            The solutions are derived from the quartic equation associated with the
            quadratic constraint hit time (Eqns 2.48–2.53 in the reference). These expressions
            were computed symbolically in Mathematica, exported to C, and then optimized
            for performance by eliminating redundant calculations.

            Memory management is handled automatically.
        )pbdoc"
    );
}
