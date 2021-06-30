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

#    if sigma_I != sigma_I0:
    term1 = 1/a**2 - 1/b**2
    term2 = 2 * (d/b**2 - c/a**2)
    term3 = c**2/a**2 - d**2/b**2 - 2 * np.log(b/a*f/e)
    I = (-term2 + np.sqrt(term2 **2 - 4 * term1 * term3))/(2*term1)
#        I = (((c/a**2) - (d/b**2)) + sqrt((d/b**2 - c/a**2)**2 - (1/a**2-1/b**2)*(c**2/a**2 - 2*log(b/a*e/f)-d**2/b**2)))/(1/a**2 - 1/b**2)
        
#    else:
#        I = 0.5 * (mu_I + mu_I0) - sigma_I ** 2 * log(f / e) / (mu_I - mu_I0)

    return I

def calculating_mu_I(data):
    """calculating mean value of diagnostic index from data"""
    # default value is set from the previous paper
    return data.mean()

def calculating_sigma_I(data):
    """calculating standard deviation of diagnostic index from data"""
    # default value is set from the previous paper
    return np.std(data)

def calculating_emission(data):
    return data.mean()

def calculating_total_cost(C_alpha, C_beta, mu_I0, sigma_I0, mu_I, sigma_I, I_thr):
    alpha = []
    beta = []
    for _mu_I0, _sigma_I0, _mu_I, _sigma_I in zip(mu_I0, sigma_I0, mu_I, sigma_I):
        alpha.append(norm(loc = _mu_I0, scale = _sigma_I0).cdf(I_thr))
        beta.append(1 - norm(loc = _mu_I, scale = _sigma_I).cdf(I_thr))    
    C_tot = np.atleast_2d(alpha) * C_alpha + np.atleast_2d(beta) * C_beta
    return C_tot

def calculating_optimal_total_cost(C_alpha, C_beta, mu_I0, sigma_I0, mu_I, sigma_I):
    alpha = []
    beta = []
    _mu_I0 = mu_I0
    _sigma_I0 = sigma_I0
    _mu_I = mu_I
    _sigma_I = sigma_I

#   for _mu_I0, _sigma_I0, _mu_I, _sigma_I in zip(mu_I0, sigma_I0, mu_I, sigma_I):
    I_thr = optimal_threshold(_mu_I, _mu_I0, _sigma_I, _sigma_I0, C_alpha, C_beta)
    alpha_cal = norm(loc = _mu_I0, scale = _sigma_I0).cdf(I_thr)
    beta_cal = 1 - norm(loc = _mu_I, scale = _sigma_I).cdf(I_thr)
    alpha.append(alpha_cal)
    beta.append(beta_cal)   
    C_tot = np.atleast_2d(alpha) * C_alpha + np.atleast_2d(beta) * C_beta
    
    return C_tot.T