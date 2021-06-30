# %%
from scipy.stats.stats import sigmaclip
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from random import choices
import random
# %%
'''
Start to SVM optimization 
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
from functools import partial
from skopt import gp_minimize, forest_minimize
from copy import deepcopy
from pyDOE2 import *
# %%

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

def func(point):
    global svm_classifier, X
    result = ( abs(value_prediction_svm(svm_classifier, point))*1e5 
                + 1/X.shape[0] * svm_uncertainty(svm_classifier, X, point))[0]
#    result = abs(value_prediction_svm(svm_classifier, point))[0]
    return result

def check_around_point(training_data, test_point):
    norm_cal = np.linalg.norm(training_data - test_point, axis = 1)
    for i in norm_cal:
        if i < 1e-5:
            return True
    return False

def check_class(x):
#    if (x[0] - 0.5)**2 +  (x[1] - 0.5)**2 <= 0.2 **2:
    if (x[0]-0.3)**2 + (x[1]-0.3)**2 <= 0.1**2:
        return 1
    else:
        return -1



# %%
'''
Generate X and y data
'''
np.random.seed(105)
#aa = lhs(2, samples = 10, criterion= 'center')
lhs_randomstate = 10593

#X = np.array([np.random.random(10), np.random.random(10)]).T
X = lhs(2, samples = 10, criterion= 'center', random_state=lhs_randomstate)
y = []
for _X in X:
        y.append(check_class(_X))
#y = np.random.choice([-1,1], 10)
print(X.shape)
print(y)
plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.xlim((0,1))
plt.ylim((0,1))

plt.savefig('./svm_classification_svmuncertainty_circle/initial_sampling.png')
# %%
svm_classifier = svm.SVC(kernel='rbf', C = 10000, gamma = 'auto')
# gamma = 1/n_features = 1/2
svm_classifier.fit(X,y)

# %%
xx, yy = np.meshgrid(np.linspace(0, 1, 500),
                     np.linspace(0, 1, 500))

# plot the decision function for each datapoint on the grid
Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.savefig('./svm_classification_svmuncertainty_circle/initial_boundary.png')
plt.show()

# %%
''' Classifier initial support vectors '''
plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.4)
plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], 
            marker='x')

#%%
bounds = [(0.0, 1.0), (0.0, 1.0)]

'''
Full loop start 
'''
# svm_classifier = svm.SVC(kernel='rbf', C = 10000, gamma = 'auto')
# gamma = 1/n_features = 1/2
# Initial data generation
X_initial = X.copy()
y_initial = y.copy()
#%%

# Initial setting
svm_classifier = svm.SVC(kernel='rbf', C = 1000, gamma = 'auto')
initial_svm_classifier = deepcopy(svm_classifier)
num_iter = 10
sorting = False

# For optimization of svm using the specified objective function
bounds = [(0.0, 1.0), (0.0, 1.0)]
iter = 0
n_initialization = 20

new_points = np.array([])
optimizaiton_result = {'x': np.array([]), 'y': np.array([])}
while iter < num_iter:
    # Fit svm
    svm_classifier.fit(X,y)
    opt_x = []
    opt_fun = []
    opt_success = []
    # optimize objective function using random sampling\
    for i in range(n_initialization):
        np.random.seed(i**2)
        opt = minimize(func, x0 = np.random.rand(2), method = "L-BFGS-B", bounds=bounds)
        opt_x.append(opt.x)
        opt_fun.append(opt.fun)
        opt_success.append(opt.success)
    
    # get the new x and y
    max_uncertainty = 0.0
    new_fun = min(opt_fun)
    new_x = opt_x[np.argmin(opt_fun)]
    if sorting:
        for idx, _x in enumerate(opt_x):
            fun_value = opt_fun[idx]
            if fun_value <= 1e-3:
                uncertainty = svm_uncertainty(svm_classifier, X, _x)
                if max_uncertainty < uncertainty:
                    max_uncertainty = uncertainty
                    new_fun = fun_value
                    new_x = _x

    # while True:
    #     new_index = np.random.randint(0,len(new_x))
    #     new_x = opt_x[new_index]        
    #     new_fun = opt_fun[new_index]
    #     if ~check_around_point(X, new_x):
    #         break

    X = np.vstack([X, new_x])
    if iter == 0 :
        new_points = new_x
    else:
        new_points = np.vstack([new_points, new_x])
    y.append(check_class(new_x))
    print('Iteration {} : Added point x value is {} and function value is {}'.format(iter, new_x, new_fun))
    iter += 1

# %%
plt.scatter(X[:10,0], X[:10,1], c=y[:10], s=30, alpha = 0.3)
#plt.plot(x,y_hyperplane)
plt.scatter(new_points[:,0], new_points[:,1], s=50, c = 'r', marker = '*')
plt.scatter(svm_classifier.support_vectors_[:,0], 
            svm_classifier.support_vectors_[:,1], 
            s=15, marker='x')

plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))
plt.legend(['Initial points', 'new points', 'support vectors'])
plt.savefig('./svm_classification_svmuncertainty_circle/final_sampling.png')
# %%
# only valid for linear hyperplane
if svm_classifier.kernel == "linear":
    x = np.linspace(0,1,100) 
    y_hyperplane = []
    for _x in x:
        y_hyperplane.append((-svm_classifier.intercept_ - svm_classifier.coef_[0][0] * _x) / svm_classifier.coef_[0][1])

    plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
    plt.scatter(svm_classifier.support_vectors_[:,0], svm_classifier.support_vectors_[:,1], marker='x')
    plt.plot(x,y_hyperplane)
    plt.ylim((-0.1,1.1))

# %%
np.random.seed(350)
test_X = np.array([np.random.random(1000), np.random.random(1000)]).T
test_y = []
for _X in test_X:
    test_y.append(check_class(_X))

print('Final svm classifier score: ', svm_classifier.score(test_X, test_y))
initial_svm_classifier.fit(X_initial, y_initial)
print('Initial svm classifier score: ', initial_svm_classifier.score(test_X, test_y))
# %%
xx, yy = np.meshgrid(np.linspace(0, 1, 500),
                     np.linspace(0, 1, 500))

# plot the decision function for each datapoint on the grid
Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.savefig('./svm_classification_svmuncertainty_circle/final_boundary.png')
plt.show()

# %%
'''Plot only LHS sampling '''
#X = np.array([np.random.random(10), np.random.random(10)]).T
X = lhs(2, samples = 20, criterion= 'center', random_state=29)
y = []
for _X in X:
        y.append(check_class(_X))
#y = np.random.choice([-1,1], 10)
print(X.shape)
print(y)
plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.xlim((0,1))
plt.ylim((0,1))
plt.savefig('./svm_classification_svmuncertainty_circle/only_lhs_sampling.png')
# %%
svm_classifier = svm.SVC(kernel='rbf', C = 10000, gamma = 'auto')
# gamma = 1/n_features = 1/2
svm_classifier.fit(X,y)

# %%
xx, yy = np.meshgrid(np.linspace(0, 1, 500),
                     np.linspace(0, 1, 500))

# plot the decision function for each datapoint on the grid
Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.savefig('./svm_classification_svmuncertainty_circle/only_lhs_boundary.png')
plt.show()

# %%