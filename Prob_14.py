import numpy as np
import matplotlib.pyplot as plt

def f(t, y, y_prime):
    return (t**3) * np.log(t) / (t**2) + (2*t*y_prime - 2*y) / (t**2)

def euler_method(t0, y0, y_prime0, h, T):
    num_steps = int((T - t0) / h) + 1
    t_values = np.linspace(t0, T, num_steps)
    y_values = np.zeros(num_steps)
    y_prime_values = np.zeros(num_steps)

    y_values[0] = y0
    y_prime_values[0] = y_prime0

    for i in range(1, num_steps):
        y_prime_values[i] = y_prime_values[i-1] + h * f(t_values[i-1], y_values[i-1], y_prime_values[i-1])
        y_values[i] = y_values[i-1] + h * y_prime_values[i-1]

    return t_values, y_values

def exact_solution(t):
    return (7*t/4) + (t**3/2) * np.log(t) - (3/4) * (t**3)

# Initial conditions and step size
t0 = 1
y0 = 1
y_prime0 = 0
h = 0.001
T = 2

# Solve using Euler's method
t_values, y_values = euler_method(t0, y0, y_prime0, h, T)

# Calculate exact solution
exact_values = exact_solution(t_values)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='Euler\'s Method')
plt.plot(t_values, exact_values, label='Exact Solution')
plt.title('Numerical and Exact Solutions')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.show()

