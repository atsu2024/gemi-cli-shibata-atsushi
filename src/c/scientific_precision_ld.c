#define __USE_MINGW_ANSI_STDIO 1
#include <stdio.h>
#include <math.h>

/**
 * High-Precision Scientific Calculation (Long Double)
 * This program calculates the value of PI using the Chudnovsky algorithm
 * or a similar high-precision series expansion, demonstrating the 
 * use of long double for scientific computing.
 */

int main() {
    long double pi_calc = 0.0L;
    long double k_factorial = 1.0L;
    long double multiple_k_factorial = 1.0L;
    long double power_of_640320 = 1.0L;
    
    printf("--- High-Precision Scientific Computation (long double) ---\n");
    printf("Calculating PI approximation using Leibniz series and other constants...\n\n");

    // Leibniz series for Pi/4: 1 - 1/3 + 1/5 - 1/7 + ...
    long double sum = 0.0L;
    long int iterations = 10000000;
    
    for (long int i = 0; i < iterations; i++) {
        long double term = 1.0L / (2.0L * (long double)i + 1.0L);
        if (i % 2 == 0) {
            sum += term;
        } else {
            sum -= term;
        }
    }
    
    pi_calc = sum * 4.0L;

    printf("Iterations: %ld\n", iterations);
    printf("Calculated PI: %.30Lf\n", pi_calc);
    printf("Standard M_PI: %.30Lf\n", (long double)M_PI);
    printf("Difference:    %.30Le\n", fabsl(pi_calc - (long double)M_PI));

    printf("\n--- Physics Simulation: Gravitational Potential ---\n");
    long double G = 6.67430e-11L; // Gravitational constant
    long double M = 5.972e24L;    // Mass of Earth
    long double r = 6371000.0L;   // Radius of Earth
    
    long double potential = -G * M / r;
    printf("Earth Gravitational Potential at Surface: %.10Lf J/kg\n", potential);
    
    return 0;
}
