#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt

def ode(x, y):
    return np.vstack((y[1], -np.exp(-2*y[0])))

def bc(ya, yb):
    return np.array([ya[0], yb[0]-np.log(2)])

x = np.linspace(1, 2, 100)
y_guess = np.zeros((2, x.size))
sol = solve_bvp(ode, bc, x, y_guess)

plt.plot(sol.x, sol.y[0], label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solution of y\'\' = -exp(-2y)')
plt.show()


# In[6]:


def ode(x, y):
    return np.vstack((y[1], y[1]*np.cos(x) - y[0]*np.log(y[0] + 1e-10)))

def bc(ya, yb):
    return np.array([ya[0]-1, yb[0]-np.e])

x = np.linspace(0, np.pi/2, 100)
y_guess = np.zeros((2, x.size))
sol = solve_bvp(ode, bc, x, y_guess)

plt.plot(sol.x, sol.y[0], label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solution of y\'\' = y\' cos(x) - y ln(y)')
plt.show()


# In[3]:


def ode(x, y):
    return np.vstack((y[1], -(2*y[1]**3 + y[0]**2 * y[1]) * np.cos(x)))

def bc(ya, yb):
    return np.array([ya[0] - 2**(-1/4), yb[0] - (12**0.25)/2])

x = np.linspace(np.pi/4, np.pi/3, 100)
y_guess = np.zeros((2, x.size))
sol = solve_bvp(ode, bc, x, y_guess)

plt.plot(sol.x, sol.y[0], label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solution of y\'\' = -(2(y\')^3 + y^2 y\') sec(x)')
plt.show()


# In[5]:


def ode(x, y):
    return np.vstack((y[1], 0.5 - 0.5*y[1]**2 - y[0]*np.sin(x)/2))

def bc(ya, yb):
    return np.array([ya[0] - 2, yb[0] - 2])

x = np.linspace(0, np.pi, 100)
y_guess = np.zeros((2, x.size))
sol = solve_bvp(ode, bc, x, y_guess)

plt.plot(sol.x, sol.y[0], label='y(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Solution of y\'\' = 1/2 - (y\')^2/2 - y sin(x)/2')
plt.show()


# In[ ]:




