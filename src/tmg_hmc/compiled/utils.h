#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

double soln1(double q1, double q2, double q3, double q4, double q5);

double soln2(double q1, double q2, double q3, double q4, double q5);

double soln3(double q1, double q2, double q3, double q4, double q5);

double soln4(double q1, double q2, double q3, double q4, double q5);

double soln5(double q1, double q2, double q3, double q4, double q5);

double soln6(double q1, double q2, double q3, double q4, double q5);

double soln7(double q1, double q2, double q3, double q4, double q5);

double soln8(double q1, double q2, double q3, double q4, double q5);

// Setup Python ctypes interface for calc_all_solutions
extern "C" {
    // double* calc_all_solutions_py(double q1, double q2, double q3, double q4, double q5){
    //     double* solutions = calc_all_solutions(q1, q2, q3, q4, q5);
    //     return solutions;
    // }
    double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5);
    
    void free_ptr(double* arr){
        free(arr);
    }
}
