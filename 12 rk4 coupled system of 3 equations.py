#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def f(t, u):
    u1, u2, u3 = u
    return np.array([
        u1 + 2*u2 - 2*u3 + np.exp(-t),
        u2 + u3 - 2*np.exp(-t),
        u1 + 2*u2 + np.exp(-t)
    ])

def runge_kutta4_step(t, u, dt, f):
    k1 = dt * f(t, u)
    k2 = dt * f(t + 0.5*dt, u + 0.5*k1)
    k3 = dt * f(t + 0.5*dt, u + 0.5*k2)
    k4 = dt * f(t + dt, u + k3)
    return u + (k1 + 2*k2 + 2*k3 + k4) / 6.0

# Initial conditions
t0 = 0
u0 = np.array([3, -1, 1])

# Time grid
N = 100
t_values = np.linspace(t0, 1, N)
dt = t_values[1] - t_values[0]

# Solve using Runge-Kutta 4th order
u_values = np.zeros((N, 3))
u_values[0] = u0
for i in range(1, N):
    u_values[i] = runge_kutta4_step(t_values[i-1], u_values[i-1], dt, f)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(t_values, u_values[:, 0], label='u1')
plt.plot(t_values, u_values[:, 1], label='u2')
plt.plot(t_values, u_values[:, 2], label='u3')
plt.xlabel('t')
plt.ylabel('u')
plt.title('Solution of the Differential Equations using Runge-Kutta 4th Order Method')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




