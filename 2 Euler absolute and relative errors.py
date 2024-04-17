#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math
import matplotlib.pyplot as plt

# Define the function dy/dt
def f(t, y):
    return y/t - (y/t)**2

# Define the analytical solution
def analytical_solution(t):
    return t / (1 + math.log(t))

# Initial values
t0 = 1
y0 = 1
h = 0.1
t_end = 20

# Initialize variables
t_values = [t0]
y_values = [y0]
absolute_errors = []
relative_errors = []
t = t0
y = y0

# Iterate using Euler's method
while t < t_end:
    y_exact = analytical_solution(t)
    y_new = y + h * f(t, y)
    y_values.append(y_new)
    t += h
    t_values.append(t)
    y = y_new
    absolute_errors.append(abs(y_exact - y_new))
    relative_errors.append(abs((y_exact - y_new) / y_exact))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label='Euler\'s Method')
plt.plot(t_values, [analytical_solution(t) for t in t_values], label='Analytical Solution')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Comparison of Analytical and Numerical Solutions')
plt.legend()
plt.grid(True)
plt.show()

# Print the errors
print("\nErrors:")
print("t\tAbsolute Error\tRelative Error")
for i in range(len(t_values)-1):
    print("{:.2f}\t{:.6f}\t{:.6f}".format(t_values[i], absolute_errors[i], relative_errors[i]))

# Plot the errors
plt.figure(figsize=(10, 6))
plt.plot(t_values[:-1], absolute_errors, label='Absolute Error')
plt.plot(t_values[:-1], relative_errors, label='Relative Error')
plt.xlabel('t')
plt.ylabel('Error')
plt.title('Absolute and Relative Errors vs t')
plt.legend()
plt.grid(True)
plt.show()
    
    
    


# In[ ]:





# In[ ]:




