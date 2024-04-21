#include <stdio.h>
#include <math.h>

// Define the function y' = f(t, y)
double f(double t, double y) {
    return y - t * t + 1;
}

// Exact solution
double exact_solution(double t) {
    return pow(t + 1, 2) - 0.5 * exp(t);
}

int main() {
    double t0 = 0;          // Initial t
    double y0 = 0.5;        // Initial y
    double h = 0.2;         // Step size
    double t, y, y_exact, error, error_bound;

    printf("t\t\tEuler's y\tExact y\t\tError\t\tError Bound\n");

    t = t0;
    y = y0;

    while (t <= 2) {
        // Euler's method
        y = y + h * f(t, y);

        // Calculate exact solution
        y_exact = exact_solution(t);

        // Calculate error
        error = fabs(y_exact - y);

        // Calculate error bound
        double M = 1.5; // Maximum of the derivative of f(t, y) over [0, 2]
        double L = 2.773; // Lipschitz constant
        error_bound = (h * M) / (2 * L) * (exp((t + h) * L) - 1);

        printf("%.2f\t\t%.6f\t%.6f\t%.6f\t%.6f\n", t, y, y_exact, error, error_bound);

        // Update t
        t += h;
    }

    return 0;
}

/*SOLUTION:
t		Euler's y	Exact y		Error		Error Bound
0.00		0.800000	0.500000	0.300000	0.040096
0.20		1.152000	0.829299	0.322701	0.109913
0.40		1.550400	1.214088	0.336312	0.231482
0.60		1.988480	1.648941	0.339539	0.443163
0.80		2.458176	2.127230	0.330946	0.811752
1.00		2.949811	2.640859	0.308952	1.453554
1.20		3.451773	3.179942	0.271832	2.571089
1.40		3.950128	3.732400	0.217728	4.516990
1.60		4.428154	4.283484	0.144670	7.905280
1.80		4.865785	4.815176	0.050608	13.805120
2.00		5.238941	5.305472	0.066531	24.078184
*/

