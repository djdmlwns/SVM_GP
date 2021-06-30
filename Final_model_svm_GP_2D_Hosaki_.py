# %%
from contextlib import suppress
from sklearn import svm
from scipy.optimize import minimize
from copy import deepcopy
from pyDOE2 import *
from statistics import * 
from math import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly 
from Test_functions import *
from sklearn.utils.optimize import _check_optimize_result
from collections import Counter

# %%
class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=2e05, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter })
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min


''' Functions definition '''
def value_prediction_svm(svm_classifier, point):    
    ''' calculate g(x) of point '''
    value_prediction = svm_classifier.decision_function(np.atleast_2d(point))   
    return value_prediction

def check_class(x):
    ''' check classification of data x '''
    fun = Hosaki2d(x)
    if fun <= -1.0:
        return 1
    else:
        return -1

def check_close_points(X, point):
    distance = X-point
    norm_set = np.linalg.norm(distance, axis = 1)

    if np.any(norm_set < 1e-4):
        return True
    else:
        return False    

def plot_heatmap_uncertainty(Gaussian_regressor):
    ''' plot heat map of uncertainty calculated by Gaussian regressor '''
    n_points = 10
    x1 = np.linspace(0,1,n_points)
    x2 = np.linspace(1,0,n_points)

    for i, _x2 in enumerate(x2):
        y_value = []
        for _x1 in x1:
            y_value.append(Gaussian_regressor.predict(np.atleast_2d([_x1,_x2]), return_std = True)[1][0])
        
        if i == 0:
            heatmap_data = np.array(y_value).reshape(1,n_points)
        else:
            heatmap_data = np.vstack([heatmap_data, np.array(y_value).reshape(1,n_points)])
#    print(heatmap_data)    
    sn.heatmap(heatmap_data)
    plt.show()

def plot_svm_boundary(svm_classifier, X, y):
    ''' plot svm decision boundary of svm_classifier with data X and y '''
    xx, yy = np.meshgrid(np.linspace(0, 1, 500),
                        np.linspace(0, 1, 500))

    # plot the decision function for each datapoint on the grid
    Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='equal',
            origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                        linestyles='dashed')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    #plt.savefig('./svm_classification_svmuncertainty_circle/final_boundary.png')
    plt.show()

def plot_scatter_data(svm_classifier, X,y, new_points):
    ''' scatter plot for data '''
    plt.scatter(X[:initial_sample_number,0], X[:initial_sample_number,1], c=y[:initial_sample_number], s=30, alpha = 0.3)
    #plt.plot(x,y_hyperplane)
    plt.scatter(new_points[:,0], new_points[:,1], s=50, c = 'r', marker = '*')
#    plt.scatter(svm_classifier.support_vectors_[:,0], 
#                svm_classifier.support_vectors_[:,1], 
#                s=15, marker='x')

    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.legend(['Initial points', 'new points', 'support vectors'])
    plt.show()

def test_svm(num_test_points, svm_classifier, initial_classifier):
    ''' test final svm accuracy and compare with initial svm '''
    global dim
    max_itr = 5
    score_lst = []
    itr = 0
    while itr < max_itr:
        test_X = np.random.random([num_test_points, dim])
        test_y = []
        for _X in test_X:
            test_y.append(check_class(_X))

#        initial_classifier.fit(X_initial, y_initial)
    #    print('Score for initial SVM is: ', initial_classifier.score(test_X, test_y))
    #    print('Score for final SVM is:', svm_classifier.score(test_X, test_y))
        
#        score_lst.append(svm_classifier.score(test_X, test_y))
#       If F1 score is needed
        prediction = svm_classifier.predict(test_X)
        Correct = prediction == test_y
        Incorrect = prediction != test_y
        Positive = test_y == np.ones(len(test_y))
        Negative = test_y == -np.ones(len(test_y))

        TP = Counter(Correct & Positive)[True]  
        TN = Counter(Correct & Negative)[True] 
        FP = Counter(Incorrect & Negative)[True]
        FN = Counter(Incorrect & Positive)[True]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_score = 2 * precision * recall / (precision + recall)
        score_lst.append(F1_score)

        itr += 1

    return mean(score_lst)

#%%
def fun(point):
    ''' Function to be minimized '''    
    global svm_classifier, Gaussian_regressor, C1
    
    # g(x) 
    fun_val = abs((value_prediction_svm(svm_classifier, point))[0]) 
    
    # U(x) 
    uncertainty = Gaussian_regressor.predict(np.atleast_2d(point), return_std = True)[1][0] 

#   print('length scale {}'.format(Gaussian_regressor.kernel_.get_params()))
    # g(x) - C1*U(x)
    return fun_val - C1 * uncertainty

def corner_addition(X, dim):
    ''' add corner points to data X '''
    add = ff2n(dim)
    add[add == - 1] = 0
    if X.size == 0 :
        return add   
    else:     
        return np.vstack([X, add])

def initial_point_sampling(dim, num_samples, method = 'doe'):
    if method == 'doe':
        return corner_addition(np.array([]), dim)
    elif method == 'lhs':
        return lhs(dim, num_samples)
    elif method == 'random':
        return np.random.random([num_samples, dim])


def mainloop(svm_classifier, num_iter, bounds, n_optimization, 
                X, y, report_frq):
    global Gaussian_regressor # generate Gaussian regressor from the loop and go to outside

    iter = 0
    score_list = []
    num_iter_list = []

    while iter < num_iter:
        # Fit svm
        svm_classifier.fit(X,y)

        # Calculate g(x) using svm
        continuous_y = svm_classifier.decision_function(X)

        kernel = ConstantKernel(constant_value=3.0, constant_value_bounds=(0.01, 10.0)) \
                 * RBF(length_scale=sqrt(X.var()), length_scale_bounds='fixed') # variance will make the length scale of SVM and GP same
        Gaussian_regressor = MyGPR(kernel = kernel, normalize_y = True) 
#        Gaussian_regressor = GaussianProcessRegressor(normalize_y = True) 
        # Train gaussian process regressor using X and continous y
        Gaussian_regressor.fit(X, continuous_y) 
#        print('Length scale is {} \n'.format(Gaussian_regressor.kernel_.get_params()['k2__length_scale']))
        if ((iter+1) % report_frq == 0) or (iter == 0):    
            np.random.seed()
            score = test_svm(1000, svm_classifier, initial_svm_classifier)
            print('Current score is {} \n'.format(score))
            score_list.append(score)
            num_iter_list.append(iter)

        # if we want to see heatmap of uncertainty, we can activate the next line
        # plot_heatmap_uncertainty(Gaussian_regressor)

        # Optimize |g(x)| - C1 * uncertainty
        opt_x = []
        opt_fun = []
        for i in range(n_optimization):
            np.random.seed()
            opt = minimize(fun, x0 = np.random.rand(dim), method = "L-BFGS-B", bounds=bounds)
            opt_x.append(opt.x)
            opt_fun.append(opt.fun)
        new_fun = min(opt_fun)
        new_x = opt_x[np.argmin(opt_fun)]
        
        if check_close_points(X, new_x):
            print('There is a similar point')
        else:

            # Add new_x to the training data
            X = np.vstack([X, new_x])
            if iter == 0 :
                new_points = np.atleast_2d(new_x)
            else:
                new_points = np.vstack([new_points, new_x])
            y.append(check_class(new_x))

            # Print
            np.set_printoptions(precision=3, suppress=True)
            print('Iteration {0} : Added point x value is {1} and function value is {2:2.2E}'.format(iter, new_x, new_fun))

        # Add iteration number
        iter += 1
    return score_list, num_iter_list, X

def LHS_LHS_sample_SVM(max_itr, step_size, iteration):
    ''' 
    Starting LHS + random sampling-based SVM 
    '''
    score_random_list = []
    for _num_iter in np.arange(0, max_itr+step_size, step_size):
        random_score_lst = []
        lhs_itr = 0
        while lhs_itr < iteration:
            if _num_iter != 0:
                X_lhs = lhs(dim, samples= _num_iter)
                X_random = np.vstack([X_initial, X_lhs])
            else: 
                X_random = X_initial

            y_random = []
            for _X in X_random:
                y_random.append(check_class(_X))

#            X_random_initial = X_random.copy()
#            y_random_initial = y_random.copy()

            # Initial setting
            random_svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42)
            random_initial_svm_classifier = deepcopy(random_svm_classifier)
            random_svm_classifier.fit(X_random,y_random)

            # Test
            random_score = test_svm(1000, random_svm_classifier, random_initial_svm_classifier)
            
            random_score_lst.append(random_score)
            
            lhs_itr += 1
        score_random_list.append(random_score_lst)
    return score_random_list

def LHS_Random_sample_SVM(max_itr, step_size, iteration):
    ''' 
    Starting random sampling-based SVM 
    '''
    score_trueran_list = []
    # True random
    for _num_iter in np.arange(0,max_itr+step_size,step_size):
        random_score_lst = []
        lhs_itr = 0
        while lhs_itr < iteration:
            np.random.seed()
            if _num_iter == 0:
                X_trueran = X_initial
            else:
                X_ran = np.random.random([_num_iter, dim])
                X_trueran = np.vstack([X_initial, X_ran])

            y_trueran = []
            for _X in X_trueran:
                y_trueran.append(check_class(_X))

            X_trueran_initial = X_trueran.copy()
            y_trueran_initial = y_trueran.copy()

            # Initial setting
            trueran_svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42)
            trueran_initial_svm_classifier = deepcopy(trueran_svm_classifier)
            trueran_svm_classifier.fit(X_trueran,y_trueran)

            # Test
            random_score = test_svm(1000, trueran_svm_classifier, trueran_initial_svm_classifier)
            
            random_score_lst.append(random_score)
            
            lhs_itr += 1
        score_trueran_list.append(random_score_lst)
    return score_trueran_list

def plot_progress_plot(num_iter_list, X_initial, score_multi_list, score_random_list, score_trueran_list, title):

    extended_num_iter_list = np.array(num_iter_list) + X_initial.shape[0]

    plt.fill_between(extended_num_iter_list, np.max(score_multi_list, axis=0), np.min(score_multi_list, axis=0), alpha=0.3, color = 'g')
    plt.scatter(extended_num_iter_list, np.mean(score_multi_list, axis=0), color='g')
    plt.plot(extended_num_iter_list, np.mean(score_multi_list, axis=0), color='g', label='optimization')

    plt.fill_between(extended_num_iter_list, np.max(score_random_list, axis=1), np.min(score_random_list, axis=1), alpha = 0.1, color='r')
    plt.scatter(extended_num_iter_list, np.mean(score_random_list, axis=1), color='r')
    plt.plot(extended_num_iter_list, np.mean(score_random_list, axis=1), color = 'r', label = 'LHS')

    plt.fill_between(extended_num_iter_list, np.max(score_trueran_list, axis=1), np.min(score_trueran_list, axis=1), alpha = 0.1, color='b')
    plt.scatter(extended_num_iter_list, np.mean(score_trueran_list, axis=1), color='b')
    plt.plot(extended_num_iter_list, np.mean(score_trueran_list, axis=1), color = 'b', label = 'Random')

    plt.title(title)
    plt.xlabel('Samples')
    plt.ylabel('SVM accuracy')
    plt.legend()
    plt.show()


# %%
################################################################################################
# Everything in the Loop

'''
Generate X and y data
'''
numpy_randomstate = 3       # set numpy random state if needed
random_state = 11           # random state for lhs sampling

# Check data shape
dim = 2
num_samples = 6 # number of initial samples

# Check data validity
X = initial_point_sampling(dim, num_samples, 'lhs')
initial_sample_number = X.shape[0]  # number of initial sampling points

y = []
for _X in X:
    y.append(check_class(_X))

if 1 in y and -1 in y:
    print('Data contains both classifications. Good to go')    
else: 
    raise ValueError('One classification data is missing. Different random state for lhs is needed.')

# copy data for plotting
X_initial = X.copy()
y_initial = y.copy()
#%%
'''
Full loop start if initial data has all classifications (-1 and 1)
'''

max_itr = 200                # number of additional samples
C1 = 1e4                   # weight on the uncertainty
report_frq = 5

# Initial parameter setting
svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42) # initial svm
initial_svm_classifier = deepcopy(svm_classifier) # copy initial svm for comparison
n_optimization = 10 # number of initialization for optimization
new_points = np.array([]) # initialize new points collection
# Variable bound
bounds = []
for i in range(dim):
    bounds.append((0.0, 1.0))   # variable bounds

# Loop start
score_multi_list = []
for inner_itr in range(5):
    y = y_initial.copy()
    score_list, num_iter_list, X_final = mainloop(svm_classifier = svm_classifier,
            num_iter = max_itr, bounds = bounds, n_optimization = n_optimization, X = X, y=y, report_frq = report_frq)
    score_multi_list.append(score_list)

# %%
# LHS + LHS Sampling-based SVM
score_random_list = LHS_LHS_sample_SVM(max_itr = max_itr, step_size = report_frq, iteration = 5)

# LHS + Random sampling-based SVM
score_trueran_list = LHS_Random_sample_SVM(max_itr = max_itr, step_size= report_frq, iteration = 5)
# %%
# Save results
#np.save("final_result_opt_200_hett4d_lhsinitial_C1_001.npy", np.array(score_multi_list))
#np.save("final_result_lhs_200_hett4d_lhsinitial_C1_001.npy", np.array(score_random_list))
#np.save("final_result_ran_200_hett4d_lhsinitial_C1_001.npy", np.array(score_trueran_list))
#%%
title = 'Hosaki 2D with C1=' + str(C1) + ' and GP using SVM length scale and F1 score'
plot_progress_plot(num_iter_list= num_iter_list, X_initial = X_initial, score_multi_list=score_multi_list,
                    score_random_list = score_random_list, score_trueran_list= score_trueran_list,
                    title = title)

# %%
# Plot svm boundary
plot_svm_boundary(svm_classifier, X_final, y)

# %%