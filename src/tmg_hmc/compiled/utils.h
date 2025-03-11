#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

// Setup Python ctypes interface for calc_all_solutions
extern "C" {
    double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5);
    
    void free_ptr(double* arr){
        delete[] arr;
    }
}