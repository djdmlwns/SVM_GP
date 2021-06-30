# %%
'''
This file has theta value optimization 
'''
import numpy as np
from statistics import * 
from scipy.stats import norm
from math import *
from auxiliary_function import *

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.stats import norm
#%%

def fun(theta, robust=False):
    # generate true distribution
    # for a, we want to have mu=0.5*theta and sigma=0.1*theta
    # for b, we want to have mu=1.0*theta and sigma=0.15*theta 
    np.random.seed(42)
    num_samples = 10
    true_mu_I = [0.8] 
    true_sigma_I = 0.1 - 0.01*(theta - 1)**2 
    true_mu_I0 = [1.0] 
    true_sigma_I0 = 0.15 * (theta - 1)**2
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

    return np.atleast_2d(sample_optimal_cost)
# %%
X_plot = np.atleast_2d(np.linspace(0.0, 1.5, 30)).T
Y_plot = fun(X_plot)
lines = []
fig = plt.figure(figsize=[5,5])
ax = fig.add_subplot(111)
true_fun = ax.plot(X_plot,Y_plot)
#lines.append(true_fun)
#ax.set_title('$x \sin{x}$ function')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
plt.show()

# %%
from smt.surrogate_models import KRG, QP

# expected improvement
def EI(GP,points,f_min):
    pred = GP.predict_values(points)
    var = GP.predict_variances(points)
    args0 = (f_min - pred)/np.sqrt(var)
    args1 = (f_min - pred)*norm.cdf(args0)
    args2 = np.sqrt(var)*norm.pdf(args0)

    if var.size == 1 and var == 0.0:  # can be use only if one point is computed
        return 0.0
    
    ei = args1 + args2
    return ei

# surrogate Based optimization: min the Surrogate model by using the mean mu
def SBO(GP,point):
    res = GP.predict_values(point)
    return res

# upper confidence bound optimization: minimize by using mu - 3*sigma
def UCB(GP,point):
    pred = GP.predict_values(point)
    var = GP.predict_variances(point)
    res = pred-3.*np.sqrt(var)
    return res
# %%
#EI, SBO, UCB are available
IC = 'SBO'

import matplotlib.image as mpimg
import matplotlib.animation as animation
from IPython.display import HTML

plt.ioff()

x_data = np.atleast_2d([0.0, 0.7, 1.5]).T
initial_sample_size = x_data.size

y_data = fun(x_data)

n_iter = 10 # number of main loop iteration
ndim = 1
# surrogate model
gpr = KRG(theta0=[1e-2]*ndim, print_global = False)
#gpr = QP(print_global = False)

# %%
for k in range(n_iter):
    # starting points
    # pick 5 points within [0,1.5] for optimization
    # this is for multi-start local optimization
    x_start = np.atleast_2d(np.random.rand(5)*1.5).T
    # find the minimum function value
    f_min_k = np.min(y_data)
    # training surrogate model with the data
    gpr.set_training_values(x_data,y_data)
    gpr.train()
    # define objective function for the optimization
    # maximize expected improvement
    if IC == 'EI':
        obj_k = lambda x: -EI(gpr,np.atleast_2d(x),f_min_k)[:,0]
    # minimize surrogate based function value
    elif IC =='SBO':
        obj_k = lambda x: SBO(gpr,np.atleast_2d(x))
    # minimize upper confidence limit
    elif IC == 'UCB':
        obj_k = lambda x: UCB(gpr,np.atleast_2d(x))
    # optimize objective function using scipy
    # try 20 random points as an initial point
    opt_all = np.array([minimize(lambda x: float(obj_k(x)), x_st, method='SLSQP', bounds=[(0,1.5)]) for x_st in x_start])
    # obtain cases when the optimal solution is found
    opt_success = opt_all[[opt_i['success'] for opt_i in opt_all]]
    # obtain objective function values
    obj_success = np.array([opt_i['fun'] for opt_i in opt_success])
    # find the index of the best solution
    ind_min = np.argmin(obj_success)
    # take the result of the optimal one
    opt = opt_success[ind_min]
    # finding x values
    x_et_k = opt['x']
    # finding true function value from the optimal solution
    y_et_k = fun(np.atleast_2d(x_et_k))
    # add the optimal data into the training set for the next iteration
    y_data = np.atleast_2d(np.append(y_data,y_et_k)).T
    x_data = np.atleast_2d(np.append(x_data,x_et_k)).T
    # Plot the predicted values and variance, and expected improvement 
    Y_GP_plot = gpr.predict_values(X_plot)
    try:
        Y_GP_plot_var  =  gpr.predict_variances(X_plot)
        Y_EI_plot = -EI(gpr,X_plot,f_min_k)
    except:
        pass        
    
    fig = plt.figure(figsize=[10,10])
    ax = fig.add_subplot(111)
    # UCB and SBO will have the same scale of function value
    if IC == 'UCB' or IC == 'SBO':
        ei, = ax.plot(X_plot,Y_EI_plot,color='red')
    else:    
    # If EI is used, then it needs another axis for plot
        ax1 = ax.twinx()
        ei, = ax1.plot(X_plot,Y_EI_plot,color='red')
    # plot true function
    true_fun, = ax.plot(X_plot,Y_plot)
    # 
    data, = ax.plot(x_data[0:k+initial_sample_size],y_data[0:k+initial_sample_size],linestyle='',marker='o',color='orange')
    opt, = ax.plot(x_data[k+initial_sample_size],y_data[k+initial_sample_size],linestyle='',marker='*',color='r')
    gp, = ax.plot(X_plot,Y_GP_plot,linestyle='--',color='g')
    sig_plus = Y_GP_plot+3*np.sqrt(Y_GP_plot_var)
    sig_moins = Y_GP_plot-3*np.sqrt(Y_GP_plot_var)
    un_gp = ax.fill_between(X_plot.T[0],sig_plus.T[0],sig_moins.T[0],alpha=0.3,color='g')
    lines = [true_fun,data,gp,un_gp,opt,ei]
    ax.set_title('$x \sin{x}$ function')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(lines,['True function','Data','GPR prediction','99 % confidence','Next point to Evaluate','Infill Criteria'])
    plt.savefig('Optimisation %d' %k)
    plt.close(fig)
    
ind_best = np.argmin(y_data)
x_opt = x_data[ind_best]
y_opt = y_data[ind_best]

print('Results : X = %s, Y = %s' %(x_opt,y_opt))

fig = plt.figure(figsize=[10,10])

ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)

ims = []
for k in range(n_iter):
    image_pt = mpimg.imread('Optimisation %d.png' %k)
    im = plt.imshow(image_pt)
    ims.append([im])
    
ani = animation.ArtistAnimation(fig, ims,interval=500)
HTML(ani.to_jshtml())
# %%
