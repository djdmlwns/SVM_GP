# %%
from math import *
import numpy as np

def HARTMANN6D(x):
    ''' x needs to be array '''
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    
    A = np.array([[10., 3., 17., 3.5, 1.7, 8],
                [0.05, 10., 17., 0.1, 8., 14.],
                [3., 3.5, 1.7, 10., 17., 8.],
                [17., 8., 0.05, 10., 0.1, 14.],]
                )

    P = 1e-4 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                        [2329., 4135., 8307., 3736., 1004., 9991.],
                        [2348., 1451., 3522., 2883., 3047., 6650.],
                        [4047., 8828., 8732., 5743., 1091., 381.]])
    
    fun = - sum(alpha[i] * exp(-sum(A[i,j] * (x[j] - P[i,j])**2 for j in range(6))) for i in range(4))

    return fun 

def HARTMANN4D(x):
    ''' x needs to be array 
    mean of zero and variance of one
    '''
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    
    A = np.array([[10., 3., 17., 3.5, 1.7, 8],
                [0.05, 10., 17., 0.1, 8., 14.],
                [3., 3.5, 1.7, 10., 17., 8.],
                [17., 8., 0.05, 10., 0.1, 14.],]
                )

    P = 1e-4 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                        [2329., 4135., 8307., 3736., 1004., 9991.],
                        [2348., 1451., 3522., 2883., 3047., 6650.],
                        [4047., 8828., 8732., 5743., 1091., 381.]])
    
    fun = 1/0.839 * (1.1 - sum(alpha[i] * exp(-sum(A[i,j] * (x[j] - P[i,j])**2 for j in range(4))) for i in range(4)))

    return fun 

def Dette8d(x):
    fun = 4 * (x[0] - 2 + 8 * x[1] - 8 *x[1] ** 2 )**2 + (3 - 4 * x[1])**2 \
           + 16*sqrt(x[2] +1) * (2*x[2] - 1) **2 + sum(log(1+sum(x[j] for j in range(2,i))) for i in range(4,8)) 
    return fun        


def Rosenbrock4d(x):
    x_bar = 15 * x - 5
    fun = 1/(3.755 * 10**5) *(sum(100 * (x_bar[i] - x_bar[i+1]**2)**2 + (1-x_bar[i])**2 for i in range(2)) - 3.827 * 10 **5)
    return fun


def Branin2d(x):
    x1 = 15 * x[0] - 5
    x2 = 15 * x[1] 
    fun =  1/51.95 * ((x2 - 5.1 * x1**2 / (4 * pi **2) + 5 * x1 / pi - 6)**2 \
            + (10 - 10 / (8 * pi)) * cos(x1) - 44.81)
    return fun

def Hosaki2d(x):
    x1 = 10 * x[0]
    x2 = 10 * x[1]
    fun = (1 - 8 * x1 + 7 * x1 **2 - 7/3 * x1**3 + 1/4 * x1 **4 ) * x2 **2 * exp(-x2)
    return fun
# %%

HARTMANN6D(np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]))
# %%

HARTMANN4D(np.array([0.20169, 0.150011, 0.476874, 0.275332]))

# %%
Dette8d(np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, 0.1, 0.1]))