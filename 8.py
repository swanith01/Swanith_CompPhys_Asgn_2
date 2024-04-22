#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equations
def eq1(t, y):
    return t * np.exp(3*t) - 2*y

def eq2(t, y):
    return 1 - (t - y)**2

def eq3(t, y):
    return 1 + y/t

def eq4(t, y):
    return np.cos(2*t) + np.sin(3*t)

# Define the initial conditions and time intervals
t_span1 = (0, 1)
t_span2 = (2, 3)
t_span3 = (1, 2)
t_span4 = (0, 1)
y0_1 = 0
y0_2 = 1
y0_3 = 2
y0_4 = 1

# Solve the IVPs
sol1 = solve_ivp(eq1, t_span1, [y0_1], t_eval=np.linspace(0, 1, 100))
sol2 = solve_ivp(eq2, t_span2, [y0_2], t_eval=np.linspace(2, 3, 100))
sol3 = solve_ivp(eq3, t_span3, [y0_3], t_eval=np.linspace(1, 2, 100))
sol4 = solve_ivp(eq4, t_span4, [y0_4], t_eval=np.linspace(0, 1, 100))

# Plot the solutions
plt.figure(figsize=(14, 10))

plt.subplot(221)
plt.plot(sol1.t, sol1.y[0])
plt.title('y′ = t e^(3t) − 2y with y(0) = 0')

plt.subplot(222)
plt.plot(sol2.t, sol2.y[0])
plt.title('y′ = 1 − (t − y)^2 with y(2) = 1')

plt.subplot(223)
plt.plot(sol3.t, sol3.y[0])
plt.title('y′ = 1 + y/t with y(1) = 2')

plt.subplot(224)
plt.plot(sol4.t, sol4.y[0])
plt.title('y′ = cos 2t + sin 3t with y(0) = 1')

plt.tight_layout()
plt.show()

