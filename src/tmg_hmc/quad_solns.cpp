#include "quad_solns.h"
#include <complex>

using namespace std;

// Function to compute all 8 solutions for the quadratic constraint hit time

double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5) {
    double* solutions = new double[8];

    // Common denominators
    const double q1_sq = q1 * q1;
    const double q4_sq = q4 * q4;
    const double denom = q1_sq + q4_sq;

    // Base term
    const complex<double> base_term = -0.5 * (q1*q2 + q4*q5) / denom;

    // First sqrt term
    const complex<double> term1_a = pow(q1*q2 + q4*q5, 2) / (denom * denom);
    const complex<double> term2_a = 2.0 * (pow(q2,2) + 2*q1*q3 - q4_sq + pow(q5,2)) / (3.0 * denom);

    // Cubic term
    const complex<double> A = q1*q2 + q4*q5;
    const complex<double> B = q2*q3 - q4*q5;
    const complex<double> C = pow(q2,2) + 2*q1*q3 - q4_sq + pow(q5,2);
    const complex<double> C_sq = C * C;
    const complex<double> C_cu = C_sq * C;
    const complex<double> B_sq = B * B;
    const complex<double> A_sq = A * A;

    const complex<double> cubic_num = -12.0*B*A + 12.0*denom*(pow(q3,2) - pow(q5,2)) + C_sq;

    const complex<double> big_term = 108.0*denom*B_sq
                                    + 108.0*A_sq*(pow(q3,2) - pow(q5,2))
                                    - 36.0*B*A*C
                                    - 72.0*denom*(pow(q3,2) - pow(q5,2))*C
                                    + 2.0*C_cu;

    const complex<double> inner_sqrt = sqrt(-4.0*pow(cubic_num,3) + pow(big_term,2));
    const complex<double> cubic_root = pow(big_term + inner_sqrt, 1.0/3.0);

    const complex<double> pow2_1_3 = pow(2.0, 1.0/3.0);
    const complex<double> term3_a = pow2_1_3 * cubic_num / (3.0 * denom * cubic_root);
    const complex<double> term4_a = cubic_root / (3.0 * pow2_1_3 * denom);

    const complex<double> first_sqrt = sqrt(term1_a - term2_a + term3_a + term4_a);
    const complex<double> half_first_sqrt = first_sqrt * 0.5;

    // Second sqrt term
    const complex<double> term1_b = 2.0 * A_sq / (denom * denom);
    const complex<double> term2_b = 4.0 * C / (3.0 * denom);

    const complex<double> diff_term = (
        (-8.0*A*A_sq)/pow(denom,3)
        + 16.0*(-B)/denom
        + 8.0*A*C/pow(denom,2)
    ) / (4.0 * first_sqrt);

    const complex<double> common_expr = term1_b - term2_b - term3_a - term4_a;
    const complex<double> second_sqrt_minus = sqrt(common_expr - diff_term);
    const complex<double> second_sqrt_plus  = sqrt(common_expr + diff_term);

    const complex<double> half_second_sqrt_minus = second_sqrt_minus * 0.5;
    const complex<double> half_second_sqrt_plus  = second_sqrt_plus  * 0.5;

    // Precompute all four possible sqrt combinations (to avoid branch inside loop)
    const complex<double> args[4] = {
        base_term - half_first_sqrt - half_second_sqrt_minus, // 000: -, -, -
        base_term - half_first_sqrt + half_second_sqrt_minus, // 010: -, -, +
        base_term + half_first_sqrt - half_second_sqrt_plus,  // 100: +, +, -
        base_term + half_first_sqrt + half_second_sqrt_plus   // 110: +, +, +
    };

    // Precompute arccos for all combinations
    const double acos1 = arccos(base_term - half_first_sqrt - half_second_sqrt_minus);
    const double acos2 = arccos(base_term - half_first_sqrt + half_second_sqrt_minus);
    const double acos3 = arccos(base_term + half_first_sqrt - half_second_sqrt_plus);
    const double acos4 = arccos(base_term + half_first_sqrt + half_second_sqrt_plus);

    // Assign all 8 solutions
    solutions[0] = -acos1;
    solutions[1] =  acos1;
    solutions[2] = -acos2;
    solutions[3] =  acos2;
    solutions[4] = -acos3;
    solutions[5] =  acos3;
    solutions[6] = -acos4;
    solutions[7] =  acos4;

    return solutions;
}



// Old  Version:
// double* calc_all_solutions(double q1, double q2, double q3, double q4, double q5) {
//     double* solutions = new double[8];

//     // Calculate the common base term
//     complex<double> base_term = -0.5 * (q1 * q2 + q4 * q5) / (pow(q1, 2) + pow(q4, 2));
    
//     // Calculate the first sqrt expression (same for all functions)
//     complex<double> term1_a = pow(q1*q2 + q4*q5, 2) / pow(pow(q1, 2) + pow(q4, 2), 2);
//     complex<double> term2_a = (2 * (pow(q2, 2) + 2*q1*q3 - pow(q4, 2) + pow(q5, 2))) / (3. * (pow(q1, 2) + pow(q4, 2)));
    
//     // Complex cubic root calculation
//     complex<double> cubic_numerator = -12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
//                             12*(pow(q1, 2) + pow(q4, 2))*(pow(q3, 2) - pow(q5, 2)) + 
//                             pow(pow(q2, 2) + 2*q1*q3 - pow(q4, 2) + pow(q5, 2), 2);
    
//     // The extremely complex term inside the cubic calculation
//     complex<double> big_term = 108*(pow(q1, 2) + pow(q4, 2))*pow(q2*q3 - q4*q5, 2) + 
//                      108*pow(q1*q2 + q4*q5, 2)*(pow(q3, 2) - pow(q5, 2)) - 
//                      36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2, 2) + 2*q1*q3 - pow(q4, 2) + pow(q5, 2)) - 
//                      72*(pow(q1, 2) + pow(q4, 2))*(pow(q3, 2) - pow(q5, 2))*(pow(q2, 2) + 2*q1*q3 - pow(q4, 2) + pow(q5, 2)) + 
//                      2*pow(pow(q2, 2) + 2*q1*q3 - pow(q4, 2) + pow(q5, 2), 3);

//     // The square root inside the cubic term
//     complex<double> inner_sqrt = sqrt(-4. * pow(cubic_numerator, 3) + pow(big_term, 2));
    
//     // Complete cubic term
//     complex<double> big_term_plus_sqrt = big_term + inner_sqrt;
//     complex<double> cubic_root = pow(big_term_plus_sqrt, (1./3.));

//     complex<double> term3_a = (pow(2, (1./3.)) * cubic_numerator) / (3. * (pow(q1, 2) + pow(q4, 2)) * cubic_root);
//     complex<double> term4_a = cubic_root / (3. * pow(2, (1./3.)) * (pow(q1, 2) + pow(q4, 2)));
    
//     // Calculate the complete first sqrt expression
//     complex<double> first_sqrt_expr = term1_a - term2_a + term3_a + term4_a;
//     complex<double> first_sqrt = sqrt(first_sqrt_expr);
    
//     // Intermediate calculations for the second sqrt
//     complex<double> term1_b = (2 * pow(q1*q2 + q4*q5, 2)) / pow(pow(q1, 2) + pow(q4, 2), 2);
//     complex<double> term2_b = (4 * (pow(q2, 2) + 2*q1*q3 - pow(q4, 2) + pow(q5, 2))) / (3. * (pow(q1, 2) + pow(q4, 2)));
    
//     // Calculate the differential term
//     complex<double> diff_term1 = (-8 * pow(q1*q2 + q4*q5, 3)) / pow(pow(q1, 2) + pow(q4, 2), 3);
//     complex<double> diff_term2 = (16 * (-(q2*q3) + q4*q5)) / (pow(q1, 2) + pow(q4, 2));
//     complex<double> diff_term3 = (8 * (q1*q2 + q4*q5) * (pow(q2, 2) + 2*q1*q3 - pow(q4, 2) + pow(q5, 2))) / pow(pow(q1, 2) + pow(q4, 2), 2);
//     complex<double> diff_numerator = diff_term1 + diff_term2 + diff_term3;
//     complex<double> diff_denom = 4. * first_sqrt;
//     complex<double> diff_term = diff_numerator / diff_denom;
    
//     // Calculate the four versions of the second sqrt expression
//     // We need different versions based on the sign pattern in the original code
//     complex<double> second_sqrt_expr_1_2 = term1_b - term2_b - term3_a - term4_a - diff_term; // For functions 1, 2
//     complex<double> second_sqrt_expr_3_4 = term1_b - term2_b - term3_a - term4_a - diff_term; // For functions 3, 4
//     complex<double> second_sqrt_expr_5_6 = term1_b - term2_b - term3_a - term4_a + diff_term; // For functions 5, 6
//     complex<double> second_sqrt_expr_7_8 = term1_b - term2_b - term3_a - term4_a + diff_term; // For functions 7, 8
    
//     complex<double> second_sqrt_1_2 = sqrt(second_sqrt_expr_1_2);
//     complex<double> second_sqrt_3_4 = sqrt(second_sqrt_expr_3_4);
//     complex<double> second_sqrt_5_6 = sqrt(second_sqrt_expr_5_6);
//     complex<double> second_sqrt_7_8 = sqrt(second_sqrt_expr_7_8);
    
//     // Calculate half values for convenience
//     complex<double> half_first_sqrt = first_sqrt / 2.0;
//     complex<double> half_second_sqrt_1_2 = second_sqrt_1_2 / 2.0;
//     complex<double> half_second_sqrt_3_4 = second_sqrt_3_4 / 2.0;
//     complex<double> half_second_sqrt_5_6 = second_sqrt_5_6 / 2.0;
//     complex<double> half_second_sqrt_7_8 = second_sqrt_7_8 / 2.0;
    
//     // Calculate all 8 solutions using the correct combinations
//     // soln1: sign before acos = -, sign before first sqrt = -, sign before second sqrt = -
//     solutions[0] = -arccos(base_term - half_first_sqrt - half_second_sqrt_1_2);
    
//     // soln2: sign before acos = +, sign before first sqrt = -, sign before second sqrt = -
//     solutions[1] = arccos(base_term - half_first_sqrt - half_second_sqrt_1_2);
    
//     // soln3: sign before acos = -, sign before first sqrt = -, sign before second sqrt = +
//     solutions[2] = -arccos(base_term - half_first_sqrt + half_second_sqrt_3_4);
    
//     // soln4: sign before acos = +, sign before first sqrt = -, sign before second sqrt = +
//     solutions[3] = arccos(base_term - half_first_sqrt + half_second_sqrt_3_4);
    
//     // soln5: sign before acos = -, sign before first sqrt = +, sign before second sqrt = -
//     solutions[4] = -arccos(base_term + half_first_sqrt - half_second_sqrt_5_6);
    
//     // soln6: sign before acos = +, sign before first sqrt = +, sign before second sqrt = -
//     solutions[5] = arccos(base_term + half_first_sqrt - half_second_sqrt_5_6);
    
//     // soln7: sign before acos = -, sign before first sqrt = +, sign before second sqrt = +
//     solutions[6] = -arccos(base_term + half_first_sqrt + half_second_sqrt_7_8);
    
//     // soln8: sign before acos = +, sign before first sqrt = +, sign before second sqrt = +
//     solutions[7] = arccos(base_term + half_first_sqrt + half_second_sqrt_7_8);
    
//     return solutions;
// }

double soln1(double q1, double q2, double q3, double q4, double q5){
    double out = -acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) - 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. - 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) - 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))
              + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

double soln2(double q1, double q2, double q3, double q4, double q5){
    double out = acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) - 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. - 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) - 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))\
             + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

double soln3(double q1, double q2, double q3, double q4, double q5){
    double out = -acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) - 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. + 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) - 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))
              + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

double soln4(double q1, double q2, double q3, double q4, double q5){
    double out = acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) - 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. + 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) - 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))\
             + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

double soln5(double q1, double q2, double q3, double q4, double q5){
    double out = -acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) + 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. - 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) + 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))
              + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

double soln6(double q1, double q2, double q3, double q4, double q5){
    double out = acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) + 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. - 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) + 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))\
             + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

double soln7(double q1, double q2, double q3, double q4, double q5){
    double out = -acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) + 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. + 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) + 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))
              + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

double soln8(double q1, double q2, double q3, double q4, double q5){
    double out = acos(-0.5*(q1*q2 + q4*q5)/(pow(q1,2) + pow(q4,2)) + 
    sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
       (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) + 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))/2. + 
    sqrt((2*pow(q1*q2 + q4*q5,2))/pow(pow(q1,2) + pow(q4,2),2) - 
       (4*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) - 
       (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
            12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
            pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
        (3.*(pow(q1,2) + pow(q4,2))*pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
            108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
            36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
            72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
             (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
            2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
            sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
              pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                 (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))) - 
       pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
          108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
          36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
          72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
           (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
          2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
          sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
               12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
               pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
            pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
              108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
              36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
              72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
               (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
              2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
        (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))) + 
       ((-8*pow(q1*q2 + q4*q5,3))/pow(pow(q1,2) + pow(q4,2),3) + 
          (16*(-(q2*q3) + q4*q5))/(pow(q1,2) + pow(q4,2)) + 
          (8*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/
           pow(pow(q1,2) + pow(q4,2),2))/
        (4.*sqrt(pow(q1*q2 + q4*q5,2)/pow(pow(q1,2) + pow(q4,2),2) - 
            (2*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)))/(3.*(pow(q1,2) + pow(q4,2))) + 
            (pow(2,(1./3.))*(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                 12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                 pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2)))/
             (3.*(pow(q1,2) + pow(q4,2))*
               pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                 108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                 36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                 72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                  (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                 2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
                 sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                      12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                      pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                   pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                     108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                     36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                     72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                      (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                     2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.)))\
             + pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
               108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
               36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
               72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
               2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3) + 
               sqrt(-4*pow(-12*(q2*q3 - q4*q5)*(q1*q2 + q4*q5) + 
                    12*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2)) + 
                    pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),2),3) + 
                 pow(108*(pow(q1,2) + pow(q4,2))*pow(q2*q3 - q4*q5,2) + 
                   108*pow(q1*q2 + q4*q5,2)*(pow(q3,2) - pow(q5,2)) - 
                   36*(q2*q3 - q4*q5)*(q1*q2 + q4*q5)*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) - 
                   72*(pow(q1,2) + pow(q4,2))*(pow(q3,2) - pow(q5,2))*
                    (pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2)) + 
                   2*pow(pow(q2,2) + 2*q1*q3 - pow(q4,2) + pow(q5,2),3),2)),(1./3.))/
             (3.*pow(2,(1./3.))*(pow(q1,2) + pow(q4,2))))))/2.);
    return out;
};

// Example usage
// int main() {
//     double q1 = -1.;
//     double q2 = -2.;
//     double q3 = -3.;
//     double q4 = -4.;
//     double q5 = 5.;
    
//     // Calculate all solutions at once
//     double* solutions = calc_all_solutions(q1, q2, q3, q4, q5);

//     // Calculate each solution individually
//     vector<double> soln_ind(8);
//     clock_t start, end;
//     start = clock();

//     // for (int i=0; i<100000; i++){
//     // soln_ind[0] = soln1(q1, q2, q3, q4, q5);
//     // soln_ind[1] = soln2(q1, q2, q3, q4, q5);
//     // soln_ind[2] = soln3(q1, q2, q3, q4, q5);
//     // soln_ind[3] = soln4(q1, q2, q3, q4, q5);
//     // soln_ind[4] = soln5(q1, q2, q3, q4, q5);
//     // soln_ind[5] = soln6(q1, q2, q3, q4, q5);
//     // soln_ind[6] = soln7(q1, q2, q3, q4, q5);
//     // soln_ind[7] = soln8(q1, q2, q3, q4, q5);
//     // }
//     // end = clock();
//     // double time_taken = double(end-start)/double(CLOCKS_PER_SEC);
//     // cout << "Time taken to calculate each solution individually 100000 times: " << time_taken << " seconds" << endl;

//     clock_t start2, end2;
//     start2 = clock();

//     for (int i=0; i<100000; i++){
//     solutions = calc_all_solutions(q1, q2, q3, q4, q5);
//     }
//     end2 = clock();
//     double time_taken2 = double(end2-start2)/double(CLOCKS_PER_SEC);
//     cout << "Time taken to calculate vector solution 100000 times: " << time_taken2 << " seconds" << endl;
//     // cout << "Speedup: " << time_taken/time_taken2 << endl;

//     // Print all solutions
//     for (int i = 0; i < 8; i++) {
//         cout << "Solution vector" << (i+1) << ": " << solutions[i] << endl;
//         // cout << "Solution individual" << (i+1) << ": " << soln_ind[i] << endl;
//     }
//     // cleanup
//     delete[] solutions;
//     return 0;
// }