# %%
'''
This file has theta value optimization 
'''

from matplotlib.markers import MarkerStyle
import numpy as np
from statistics import * 
from scipy.stats import norm
from math import *
from auxiliary_function import *

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import norm
from functools import partial
from skopt import gp_minimize, forest_minimize

#%%

def fun(theta, robust=False):
    # generate true distribution
    # for a, we want to have mu=0.5*theta and sigma=0.1*theta
    # for b, we want to have mu=1.0*theta and sigma=0.15*theta 
    np.random.seed(2)
    num_samples = 5
    true_mu_I = 0.8 + 0.01 * (theta[0] - 0.2)**2
    true_sigma_I = 0.1 - 0.01*(theta[0] -0.5)**2 - 0.001 * max(exp((theta[0] - 1)), 0 )
    true_mu_I0 = 1.0 
    true_sigma_I0 = 0.15 + 0.05 * (theta[0] -0.5)**2 
    # sample the data
    a = true_sigma_I * np.random.randn(num_samples) + true_mu_I
    b = true_sigma_I0 * np.random.randn(num_samples) + true_mu_I0

    # specify requirements for the optimal threshold calculation
    C_alpha = 1
    C_beta = 1

    sigma_I = calculating_sigma_I(a)
    sigma_I0 = calculating_sigma_I(b)
    mu_I = calculating_mu_I(a)
    mu_I0 = calculating_mu_I(b)

    if robust:
        sigma_I_error = sigma_I / (sqrt(2 * num_samples - 2))
        sigma_I0_error = sigma_I0 / (sqrt(2 * num_samples - 2))
        mu_I_error = sigma_I
        mu_I0_error = sigma_I0
        sigma_I = sigma_I + sigma_I_error
        sigma_I0 = sigma_I0 + sigma_I0_error
        mu_I = mu_I + mu_I_error
        mu_I0_error = mu_I0 - mu_I0_error
    
#    sample_optimal_threshold_value = optimal_threshold(mu_I, mu_I0, sigma_I, sigma_I0, C_alpha, C_beta)
    sample_optimal_cost = calculating_optimal_total_cost(C_alpha, C_beta, mu_I0, sigma_I0, mu_I, sigma_I)

    return sample_optimal_cost[0,0]

#%%
x = np.linspace(0,1.5,30)
x_plot = []
for _x in x:
    x_plot.append([_x])

y = list(map(fun, x_plot))

plt.plot(x, y)
plt.grid(True)

#%%
func = fun
bounds = [(0.1, 1.5)]
n_calls = 30

def run1(minimizer, n_iter=1):
    return [minimizer(func, 
                    bounds, 
                    n_calls=n_calls, 
                    random_state=1,
                    noise=0.1,
                    n_initial_points=5,
                    initial_point_generator = 'lhs',
                    n_restarts_optimizer = 10, # this is only for acquisition function. not for regression
                    acq_optimizer = 'lbfgs',
                    gp_base = "RBF" # Matern, RBF, RationalQuadratic, ExpSineSquared
                    )
            for n in range(n_iter)]

def run2(minimizer, n_iter=1):
    return [minimizer(func, 
                    bounds, 
                    n_calls=n_calls, 
                    random_state=n+1)
            for n in range(n_iter)]
# Gaussian processes
gp_res = run1(gp_minimize)
rf_res = run2(forest_minimize)

print('done')

# %%
# Initial plot
x_normalize = np.linspace(0,1,100)
x_plot2d = []
for _x in x_normalize:
    x_plot2d.append(np.atleast_2d(_x))

y_plot2d_in = []
for _x in x_plot2d:
    y_plot2d_in.append(gp_res[0].models[0].predict(_x))

# final plot
y_plot2d_fin = []
for _x in x_plot2d:
    y_plot2d_fin.append(gp_res[0].models[-1].predict(_x))

# Back to original range
x_plot2d_array = ((1.5 - 0.1 ) * np.array(x_plot2d) + 0.1).squeeze(axis = 2)
plt.plot(x_plot2d_array, y_plot2d_in, 'g--')
plt.plot(x_plot2d_array, y_plot2d_fin, 'b--')
plt.plot(x, y, 'r-', alpha=0.5)
plt.scatter(np.array(gp_res[0].x_iters), gp_res[0].func_vals, s = 10)
plt.grid(True)
plt.legend(['initial', 'final',' true'])
# %%
# %%
# Convergence plot
from skopt.plots import plot_convergence

plot = plot_convergence(("gp_minimize", gp_res),
                        ("forest_minimize", rf_res))

plot.legend(loc="best", prop={'size': 6}, numpoints=1)

# %%
