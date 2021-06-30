# %%
from scipy.stats.stats import sigmaclip
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from random import choices
import random
# %%
'''
Start to SVM optimization by Bayesian
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
from copy import deepcopy
from pyDOE2 import *
# %%
'''
Generate X and y data
'''
np.random.seed(3)
#X = np.array([np.random.random(10), np.random.random(10)]).T
X = lhs(2, samples= 10, random_state=11)
y = []
for _X in X:
    if _X[0] <= 0.5:
        y.append(1)
    else:
        y.append(-1)
#y = np.random.choice([-1,1], 10)
print(X.shape)
print(y)
plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.xlim((0,1))
plt.ylim((0,1))
# %%
svm_classifier = svm.SVC(kernel='rbf', C = 10000, gamma = 'auto')
# gamma = 1/n_features = 1/2
svm_classifier.fit(X,y)

# %%
print(svm_classifier.predict(X))
print(y)
# %%
plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.4)
plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], 
            marker='x')

# %%
def value_prediction_svm(svm_classifier, point):
    dual_coef = svm_classifier.dual_coef_
    bias = svm_classifier.intercept_
    kernel = svm_classifier.kernel
    sp_vec = svm_classifier.support_vectors_
    if kernel == 'linear':
        kernel_estimation = np.dot(sp_vec, point)
    else:
        kernel_estimation = np.exp(-1/2 * np.linalg.norm(sp_vec - point, axis = 1))
    value_prediction_svm = np.dot(dual_coef, kernel_estimation) + bias    
    return value_prediction_svm

def svm_uncertainty(svm_classifier, training_data, test_point):
    if svm_classifier.get_params()['gamma'] == 'auto':
        gamma = 1/svm_classifier.shape_fit_[1]
        similarity = np.exp(-np.linalg.norm(training_data - test_point, axis = 1) * gamma).sum()
    return similarity

# %%
# only valid for linear hyperplane
x = np.linspace(0,1,100) 
y_hyperplane = []
for _x in x:
    y_hyperplane.append((-svm_classifier.intercept_ - svm_classifier.coef_[0][0] * _x) / svm_classifier.coef_[0][1])

plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], marker='x')
plt.plot(x,y_hyperplane)

# check hyperplane prediction value
hyperplane_value = []
for data in X:
    hyperplane_value.append(list(value_prediction_svm(svm_classifier, data)))

print(hyperplane_value)
print(y)

# %%


#%%

def fun(point):
    global svm_classifier
    result = abs((value_prediction_svm(svm_classifier, point))[0])  
    if point[0] < 0.01 or point[0] > 0.99 or point[1] < 0.01 or point[1] > 0.99:
        result += 5
    return result  

#%%
func = fun
bounds = [(0.0, 1.0), (0.0, 1.0)]
n_calls = 1

def run1(minimizer, x0 = None, y0 = None, n_iter=1):
    return [minimizer(func, 
                    bounds, 
                    n_calls=n_calls, 
                    random_state=1,
                    noise=0.0,
                    x0 = x0,
                    y0 = y0,
                    n_initial_points=10,
                    initial_point_generator = 'lhs',
                    n_restarts_optimizer = 10, # this is only for acquisition function. not for regression
#                    acq_optimizer = 'lbfgs',
                    acq_optimizer = 'sampling',
                    gp_base = "RBF", # Matern, RBF, RationalQuadratic, ExpSineSquared
                    acq_func = "LCB")
            for n in range(n_iter)]

# %%

plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], marker='x')
#plt.plot(x,y_hyperplane)
plt.scatter(gp_res[0].x[0], gp_res[0].x[1])
print('x value is ', gp_res[0].x, 'function value is ', gp_res[0].fun)
# %%

'''
Full loop start 
'''
# svm_classifier = svm.SVC(kernel='rbf', C = 10000, gamma = 'auto')
# gamma = 1/n_features = 1/2
# Initial data generation
np.random.seed(3)
X = np.array([np.random.random(10), np.random.random(10)]).T
y = []
for _X in X:
    if _X[0] <= 0.5:
        y.append(1)
    else:
        y.append(-1)

X_initial = X.copy()
y_initial = y.copy()
#%%

# Initial setting
svm_classifier = svm.SVC(kernel='rbf', C = 10000, gamma = 'auto')
initial_svm_classifier = deepcopy(svm_classifier)
num_iter = 20

# For bayesian optimization
func = fun
bounds = [(0.0, 1.0), (0.0, 1.0)]
n_calls = 20
iter = 0
new_points = np.array([])
while iter < num_iter:
    # Fit svm
    svm_classifier.fit(X,y)
    # Gaussian processes
    gp_res = run1(gp_minimize, x0=[list(i) for i in list(X)], y0=list(map(func, X)))
    # get the new x and y
    # get all x_iters and calculate surrogate function value with the last regressor
    # Then, estimate standard deviation among all x_iters with respect to the first regressor
    # Take the point where surrogate function value is sufficiently low and variance is max
    first_surrogate = gp_res[0].models[0]
    last_surrogate = gp_res[0].models[-1]
    func_pred_last_surrogate = last_surrogate.predict(gp_res[0].x_iters)
    std_pred_first_surrogate = first_surrogate.predict(gp_res[0].x_iters, return_std = True)[1]
    next_x = gp_res[0].x
    next_fun = gp_res[0].fun
    next_std = 0.0

    for k,(i,j) in enumerate(zip(func_pred_last_surrogate, std_pred_first_surrogate)):
        if i < 1.0:
            if j > next_std:
                next_x = gp_res[0].x_iters[k]
                next_fun = gp_res[0].func_vals[k]
                next_std = j 

    X = np.vstack([X, next_x])
    if iter == 0 :
        new_points = next_x
    else:
        new_points = np.vstack([new_points, next_x])
    y.append(1 if next_x[0] <= 0.5 else -1)
    print('Iteration {} : Added point x value is {} and function value is {}'.format(iter, next_x, next_fun))
    iter += 1

# %%
plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.3)
plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], marker='x')
#plt.plot(x,y_hyperplane)
plt.scatter(new_points[:,0], new_points[:,1], c = y[10:], marker = '*')
# %%
# only valid for linear hyperplane
x = np.linspace(0,1,100) 
y_hyperplane = []
for _x in x:
    y_hyperplane.append((-svm_classifier.intercept_ - svm_classifier.coef_[0][0] * _x) / svm_classifier.coef_[0][1])

plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], marker='x')
plt.plot(x,y_hyperplane)
plt.ylim((-0.1,1.1))

# %%
test_svm = svm.SVC(kernel = 'linear', C = 1000)
test_svm.fit(X,y)
plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.3)
plt.scatter(test_svm.support_vectors_[:,0], test_svm.support_vectors_[:,1], marker='x')
#plt.plot(x,y_hyperplane)
plt.scatter(new_points[:,0], new_points[:,1], c = y[10:], marker = '*')
# %%
# only valid for linear hyperplane
x = np.linspace(0,1,100) 
y_hyperplane = []
for _x in x:
    y_hyperplane.append((-test_svm.intercept_ - test_svm.coef_[0][0] * _x) / test_svm.coef_[0][1])

plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.scatter(test_svm.support_vectors_[:,0], test_svm.support_vectors_[:,1], marker='x')
plt.plot(x,y_hyperplane)
plt.ylim((-0.1,1.1))
# %%
np.random.seed(300)
test_X = np.array([np.random.random(100), np.random.random(100)]).T
test_y = []
for _X in test_X:
    if _X[0] <= 0.5 :
        test_y.append(1)
    else:
        test_y.append(-1)

print(svm_classifier.score(test_X, test_y))
initial_svm_classifier.fit(X_initial, y_initial)
print(initial_svm_classifier.score(test_X, test_y))
# %%




###########################################################################
# %%
''' Starting Random sampling-based SVM improvement '''
np.random.seed(3)
X = np.array([np.random.random(10), np.random.random(10)]).T
y = []
for _X in X:
    if _X[0] <= 0.5 :
        y.append(1)
    else:
        y.append(-1)

X_initial = X.copy()
y_initial = y.copy()

# Initial setting
random_svm_classifier = svm.SVC(kernel='linear', C = 1, gamma = 'auto')
random_initial_svm_classifier = deepcopy(random_svm_classifier)
num_iter = 30

# For bayesian optimization
func = fun
bounds = [(0.0, 1.0), (0.0, 1.0)]
iter = 0
new_points = np.array([])
while iter < num_iter:
    # Fit svm
    random_svm_classifier.fit(X,y)
    # get the new x and y
    np.random.seed()
    x_sample = [np.random.rand(), np.random.rand()]
    X = np.vstack([X, x_sample])
    if iter == 0 :
        new_points = x_sample
    else:
        new_points = np.vstack([new_points, x_sample])
    y.append(1 if x_sample[0] <= 0.5 else -1)
    print('Iteration {} : Added point x value is {}'.format(iter, x_sample))
    iter += 1

plt.scatter(X[:,0], X[:,1], c= 10 * np.array(y), alpha = 0.3)
#plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], marker='x')
#plt.plot(x,y_hyperplane)
plt.scatter(new_points[:,0], new_points[:,1], c=10 * np.array(y[10:]), marker = '*')

# Test
np.random.seed(300)
test_X = np.array([np.random.random(100), np.random.random(100)]).T
test_y = []
for _X in test_X:
    if _X[0] <= 0.5 :
        test_y.append(1)
    else:
        test_y.append(-1)

print(initial_svm_classifier.score(test_X, test_y))
print(random_svm_classifier.score(test_X, test_y))

# %%
from scipy.stats import norm
import numpy as np 
import matplotlib.pyplot as plt

x_norm = np.linspace(-7,7,500)
y_norm = norm(0,1).pdf(x_norm)
plt.plot(x_norm, y_norm)
y_norm = norm(1,2).pdf(x_norm)
plt.plot(x_norm, y_norm)
plt.legend(['mu=0, sigma=1', 'mu=1, sigma=2'])

# %%
