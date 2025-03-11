#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

double* A_dot_x_base(const double* x, const double* Arows, const double* Avals, const int n, const int n_comp);

double x_dot_A_dot_x_base(const double* x, const double* Arows, const double* Avals, const int n, const int n_comp);

// Setup Python ctypes interface for calc_all_solutions
extern "C" {
    double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5);

    double* A_dot_x(void* x, void* Arows, void* Avals, int n, int n_comp){
        double* x_arr = (double*) x;
        double* Arows_arr = (double*) Arows;
        double* Avals_arr = (double*) Avals;
        return A_dot_x_base(x_arr, Arows_arr, Avals_arr, n, n_comp);
    }

    double x_dot_A_dot_x(void* x, void* Arows, void* Avals, int n, int n_comp){
        double* x_arr = (double*) x;
        double* Arows_arr = (double*) Arows;
        double* Avals_arr = (double*) Avals;
        return x_dot_A_dot_x_base(x_arr, Arows_arr, Avals_arr, n, n_comp);
    }
    
    void free_ptr(double* arr){
        free(arr);
    }
}