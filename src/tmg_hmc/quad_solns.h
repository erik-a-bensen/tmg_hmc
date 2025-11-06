#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cmath>
#include <vector>
// #include <iostream>

namespace py = pybind11;

std::complex<long double> stable_acos(std::complex<long double> z) {
    // acos(z) = -i * log(z + sqrt(z^2 - 1))
    const std::complex<long double> i(0.0, 1.0);
    const std::complex<long double> one(1.0, 0.0);
    return -i * std::log(z + i * std::sqrt(one - z * z));
}

double arccos(std::complex<long double> z) {
    std::complex<long double> val = stable_acos(z);
    // Convert long double complex to double complex
    double real_part = static_cast<double>(std::real(val));
    double imag_part = static_cast<double>(std::imag(val));
    if (std::abs(imag_part) > 1e-1) {
        return std::numeric_limits<double>::quiet_NaN();  // Return NaN for significant imaginary parts
    }
    return real_part;
}

// Allow int * complex<double>
inline std::complex<double> operator*(int lhs, const std::complex<double>& rhs) {
    return static_cast<double>(lhs) * rhs;
}

// Allow complex<double> * int
inline std::complex<double> operator*(const std::complex<double>& lhs, int rhs) {
    return lhs * static_cast<double>(rhs);
}

// Allow same for complex<long double>
inline std::complex<long double> operator*(int lhs, const std::complex<long double>& rhs) {
    return static_cast<long double>(lhs) * rhs;
}

// Allow complex<long double> * int
inline std::complex<long double> operator*(const std::complex<long double>& lhs, int rhs) {
    return lhs * static_cast<long double>(rhs);
}

// Allow complex<long double> * double
inline std::complex<long double> operator*(const std::complex<long double>& lhs, double rhs) {
    return lhs * static_cast<long double>(rhs);
}

// Allow double * complex<long double>
inline std::complex<long double> operator*(double lhs, const std::complex<long double>& rhs) {
    return static_cast<long double>(lhs) * rhs;
}

// Allow complex<long double> / double
inline std::complex<long double> operator/(const std::complex<long double>& lhs, double rhs) {
    return lhs / static_cast<long double>(rhs);
}



double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5);
double soln1(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);
double soln2(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);
double soln3(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);
double soln4(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);
double soln5(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);
double soln6(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);
double soln7(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);
double soln8(std::complex<double> q1in, std::complex<double> q2in, std::complex<double> q3in, std::complex<double> q4in, std::complex<double> q5in);

/**
 * Pybind11 module for computing all 8 solutions for the full quadratic constraint hit time.
 */
PYBIND11_MODULE(compiled, m) {
    m.doc() = R"pbdoc(
        Pybind11 wrapper for the C++ quadratic constraint hit time calculations.
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
            calc_all_solutions(q1: float, q2: float, q3: float, q4: float, q5: float) -> np.ndarray

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

    m.def(
        "soln1",
        &soln1,
        R"pbdoc(
            soln1(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the first solution for the quadratic constraint hit time.

            Parameters
            ----------
            See `calc_all_solutions` for parameter descriptions.

            Returns
            -------
            float
                The computed first solution.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            The solution is derived from the quartic equation associated with the quadratic constraint hit time
            given in Eqns 2.48 - 2.53 of Pakman and Paninski (2014). The expression is computed exactly using
            Mathematica and then exported to C. See resources/HMC_exact_soln.nb for details.

            It is not recommended to use this function directly due to its complexity. Instead, use 'calc_all_solutions'
            which has been optimized to remove redundant calculations.
            This function is maintained for reference and validation purposes.
        )pbdoc"
    );

    m.def(
        "soln2",
        &soln2,
        R"pbdoc(
            soln2(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the second solution for the quadratic constraint hit time.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            See 'soln1' for details. 
        )pbdoc"
    );

    m.def(
        "soln3",
        &soln3,
        R"pbdoc(
            soln3(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the third solution for the quadratic constraint hit time.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            See 'soln1' for details. 
        )pbdoc"
    );

    m.def(
        "soln4",
        &soln4,
        R"pbdoc(
            soln4(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the fourth solution for the quadratic constraint hit time.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            See 'soln1' for details. 
        )pbdoc"
    );

    m.def(
        "soln5",
        &soln5,
        R"pbdoc(
            soln5(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the fifth solution for the quadratic constraint hit time.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            See 'soln1' for details. 
        )pbdoc"
    );

    m.def(
        "soln6",
        &soln6,
        R"pbdoc(
            soln6(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the sixth solution for the quadratic constraint hit time.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            See 'soln1' for details. 
        )pbdoc"
    );

    m.def(
        "soln7",
        &soln7,
        R"pbdoc(
            soln7(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the seventh solution for the quadratic constraint hit time.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            See 'soln1' for details. 
        )pbdoc"
    );

    m.def(
        "soln8",
        &soln8,
        R"pbdoc(
            soln8(q1: float, q2: float, q3: float, q4: float, q5: float) -> float

            Compute the eighth solution for the quadratic constraint hit time.

            Notes
            -----
            DO NOT MODIFY THIS FUNCTION
            See 'soln1' for details. 
        )pbdoc"
    );
}
