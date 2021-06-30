# %%
'''
This file is to test that it is safe to use the sampled data to construct
analytical solution for the diagnostic index.
'''
import numpy as np
from statistics import * 
from scipy.stats import norm
from math import *

def optimal_threshold(mu_I, mu_I0, sigma_I, sigma_I0, C_alpha, C_beta):
    """
    Note: when sigma_I and sigma_I0 are significantly different between I and I0, there is no solution
    
    mu_I: mean value of diagnostic index at arbitrary degradation level
    mu_I0: mean value of diagnostic index at critical degradation level
    sigma_I: variance of diagnostic index at arbitrary degradation level
    sigma_I0: variance of diagnostic index at critical degradation level
    C_alpha: cost for detection failure (type 2 error: regulation), associated with alpha
    C_beta: cost for false alarm (type 1 error: catalyst cost), associated with beta
    simple: if true, sigma_I and sigma_I0 are the same, so simplified
    """
    a = sigma_I
    b = sigma_I0
    c = mu_I
    d = mu_I0
    e = C_alpha
    f = C_beta

    if sigma_I != sigma_I0:
        term1 = 1/a**2 - 1/b**2
        term2 = 2 * (d/b**2 - c/a**2)
        term3 = c**2/a**2 - d**2/b**2 - 2 * log(b/a*f/e)
        I = (-term2 + sqrt(term2 **2 - 4 * term1 * term3))/(2*term1)
#        I = (((c/a**2) - (d/b**2)) + sqrt((d/b**2 - c/a**2)**2 - (1/a**2-1/b**2)*(c**2/a**2 - 2*log(b/a*e/f)-d**2/b**2)))/(1/a**2 - 1/b**2)
        
    else:
        I = 0.5 * (mu_I + mu_I0) - sigma_I ** 2 * log(f / e) / (mu_I - mu_I0)

    return I

def calculating_mu_I(data):
    """calculating mean value of diagnostic index from data"""
    # default value is set from the previous paper
    return data.mean()

def calculating_sigma_I(data):
    """calculating standard deviation of diagnostic index from data"""
    # default value is set from the previous paper
    return stdev(data)

def calculating_emission(data):
    return data.mean()

def calculating_total_cost(C_alpha, C_beta, mu_I0, sigma_I0, mu_I, sigma_I, I_thr):
    alpha = norm(loc = mu_I0, scale = sigma_I0).cdf(I_thr)
    beta = 1 - norm(loc = mu_I, scale = sigma_I).cdf(I_thr)
    C_tot = alpha * C_alpha + beta * C_beta
    return C_tot

def calculating_optimal_total_cost(C_alpha, C_beta, mu_I0, sigma_I0, mu_I, sigma_I):
    I_thr = optimal_threshold(mu_I, mu_I0, sigma_I, sigma_I0, C_alpha, C_beta)
    alpha = norm(loc = mu_I0, scale = sigma_I0).cdf(I_thr)
    beta = 1 - norm(loc = mu_I, scale = sigma_I).cdf(I_thr)
    C_tot = alpha * C_alpha + beta * C_beta
    return C_tot

#%%
# for a, we want to have mu=0.5 and sigma=1
true_mu_I = 0.5
true_sigma_I = 0.1
num_samples = 100
a = true_sigma_I * np.random.randn(num_samples) + true_mu_I
# for b, we want to have mu=1.0 and sigma=1.5 
true_mu_I0 = 1.0
true_sigma_I0 = 0.15
b = true_sigma_I0 * np.random.randn(num_samples) + true_mu_I0

print('a mean', mean(a))
print('a std', stdev(a))
print('b mean', mean(b))
print('b std', stdev(b))

# %%
# Testing the result from the sample mean and variance
# This robust is true when we want to consider the uncertainty in the
# distribution of the diagnostic index
robust = True

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

sample_optimal_threshold_value = optimal_threshold(mu_I, mu_I0, sigma_I, sigma_I0, C_alpha, C_beta)
sample_optimal_cost = calculating_optimal_total_cost(C_alpha, C_beta, mu_I0, 
                                                sigma_I0, mu_I, sigma_I)
print('Estimated optimal threshold is', sample_optimal_threshold_value,'\n')
print('Estimated optimal cost is', sample_optimal_cost,'\n')

# %%
# Testing the result from the population (not from sample)
C_alpha = 1
C_beta = 1
sigma_I = true_sigma_I
sigma_I0 = true_sigma_I0
mu_I = true_mu_I
mu_I0 = true_mu_I0
optimal_threshold_value = optimal_threshold(mu_I, mu_I0, sigma_I, sigma_I0, C_alpha, C_beta)
optimal_cost = calculating_optimal_total_cost(C_alpha, C_beta, mu_I0, 
                                                sigma_I0, mu_I, sigma_I)
print('True optimal threshold is', optimal_threshold_value,'\n')
print('Estimated optimal threshold is', sample_optimal_threshold_value,'\n')

print('True optimal cost is with the true optimal threshold ', optimal_cost,'\n')
robust_optimal_cost = calculating_total_cost(C_alpha, C_beta, mu_I0, sigma_I0, mu_I, 
                                            sigma_I, sample_optimal_threshold_value)
print('Estimated optimal cost is with the robust optimal', robust_optimal_cost,'\n')

# %%
