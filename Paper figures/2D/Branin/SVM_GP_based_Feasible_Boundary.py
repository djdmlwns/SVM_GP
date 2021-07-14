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

from sklearn.utils.optimize import _check_optimize_result
from collections import Counter
from Test_functions import *

# %%
''' Class Definition '''
class MyGPR(GaussianProcessRegressor):
    ''' 
    To change the maximum number of iteration of nonlinear solver within GaussianProcess Regressor
    Other settings are identical
    '''
    def __init__(self, *args, max_iter=3e6, **kwargs):
        super().__init__(*args, **kwargs)
        # To change maximum iteration number
        self._max_iter = max_iter

    # _constrained_optimization is the function for optimization inside GaussianProcessRegressor
    # Redefine this to change the default setting for the maximum iteration
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            # change maxiter option
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter })
            # _check_optimize_result is imported from sklearn.utils.optimize 
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min

#############################################################################################################
''' Function Definition '''
def fun(point):
    ''' Objective function to be minimized '''    
    global svm_classifier, Gaussian_regressor, C1
    
    # g(x) 
    fun_val = abs((value_prediction_svm(svm_classifier, point))[0]) 
    
    # U(x) 
    uncertainty = Gaussian_regressor.predict(np.atleast_2d(point), return_std = True)[1][0] 

#   print('length scale {}'.format(Gaussian_regressor.kernel_.get_params()))
    # g(x)**2 - C1*log(U(x))
    return fun_val - C1 * uncertainty


def value_prediction_svm(svm_classifier, point):    
    ''' calculate g(x) value of point '''
    value_prediction = svm_classifier.decision_function(np.atleast_2d(point))   
    return value_prediction # g(x) value


def check_class(x, func, condition):
    ''' check classification of data x '''
    # This is checked by function value for now
    # It will be determined by simulation for future use
    fun = func(x) 
    if condition(x): 
        return 1 # positive class (considered as feasible)
    else:
        return -1 # negative class (considered infeasible)
        
#################################################################################################################
def check_close_points(X, point):
    ''' To check whether there are close data around the new sample point '''
    distance = X-point
    norm_set = np.linalg.norm(distance, axis = 1) # 2-norm of distances

    if np.any(norm_set < 1e-4):
        return True 
    else:
        return False    

def test_svm(num_test_points, dim, svm_classifier, method = 'F1'):
    ''' 
    Test prediction accuracy of SVM 

    num_test_points : number of points for accuracy test

    dim : Number of features in X

    method: {'F1', 'MCC', 'Simple'}

        F1: F1-score

        MCC: Matthews correlation coefficient

        Simple: Simple accuracy (correct / total)
    '''
    # Maximum iteration for mean value
    max_itr = 5
    score_lst = []
    itr = 0
    while itr < max_itr:
        test_X = np.random.random([num_test_points, dim])
        test_y = []
        for _X in test_X:
            test_y.append(check_class(_X, func = func, condition = condition))
        prediction = svm_classifier.predict(test_X)

        # Simple accuracy
        if method == 'Simple':
            score = svm_classifier.score(test_X, test_y)
        else:            
            # Correct classification
            Correct = prediction == test_y
            # Incorrect classification
            Incorrect = prediction != test_y
            # True value is +1
            Positive = test_y == np.ones(len(test_y))
            # True value is -1
            Negative = test_y == -np.ones(len(test_y))

            TP = Counter(Correct & Positive)[True]   # True positive
            TN = Counter(Correct & Negative)[True]   # True negative
            FP = Counter(Incorrect & Negative)[True] # False positive
            FN = Counter(Incorrect & Positive)[True] # False negative
            # If method is F1-score
            if method == 'F1':
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                if (precision == 0 and recall == 0):
                    score = 0
                else:
                    score = 2 * precision * recall / (precision + recall)
            # If method is MCC
            elif method == 'MCC':
                score = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
            else:
                raise NotImplementedError('There is no such method for accuracy calculation')
        score_lst.append(score)

        itr += 1

    return mean(score_lst) 

#################################################################################################################
# Sampling methods
def corner_addition(X, dim):
    ''' 
    Auxiliary function for DOE initial sampling (Full factorial design) 
    Finding all corner points and add to X
    '''
    # import ff2n function from pyDOE2
    add = ff2n(dim)
    # default bound is [-1,1], but our bound is [0,1]
    add[add == - 1] = 0
    if X.size == 0 :
        return add   
    else:     
        return np.vstack([X, add])

def initial_point_sampling(dim, num_samples, method = 'doe'):
    '''
    Function to generate initial samples
    method: {'doe', 'lhs', 'random'}
    doe: Full factorial
    lhs: Latin Hypercube Sampling
    random: Random sampling
    '''
    if method == 'doe':
        return corner_addition(np.array([]), dim)
    elif method == 'lhs':
        # import lhs function from pyDOE2
        return lhs(dim, num_samples)
    elif method == 'random':
        return np.random.random([num_samples, dim])

#################################################################################################################
# Main Loop
def mainloop(svm_classifier, num_iter, bounds, n_optimization, 
                X, y, report_frq, accuracy_method):
    '''
    Main loop for the proposed algorithm
    
    svm_classifier: initial svm classifier (untrained)

    num_iter: number of sampling points
    
    bounds: bounds of training data X
    
    n_optimization: Number of optimization to minimize the objective function
    
    X: Initial training data (initial samples)
    
    y: Classification of initial training data
    
    report_frq: SVM accuracy printing/saving frequency

    accuracy_method : method to calculate accuracy
    
    '''
    global Gaussian_regressor # generate Gaussian regressor and send outside of the loop

    dim = X.shape[1]
    iter = 0
    # initialize score list
    score_list = []
    # initialize number of samples list
    num_iter_list = []

    while iter < num_iter:
        # Fit svm
        svm_classifier.fit(X,y)

        # Calculate g(x) using svm
        continuous_y = value_prediction_svm(svm_classifier, X)

        # Define kernel for GP (Constant * RBF)
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-3, 1e1)) \
                 * RBF(length_scale=sqrt(X.var() * X.shape[1]/2), length_scale_bounds='fixed') # Length scale is determined as same as that of SVM RBF kernel

        # Define GP
        Gaussian_regressor = MyGPR(kernel = kernel, normalize_y = True, n_restarts_optimizer = 3, alpha = 1e-5) 
#        Gaussian_regressor = MyGPR(normalize_y = True)
        # Train gaussian process regressor using X and continous y
        Gaussian_regressor.fit(X, continuous_y) 
#        print('Length scale is {} \n'.format(Gaussian_regressor.kernel_.get_params()['k2__length_scale']))

        # Test svm and append score and iteration number to list
        if ((iter+1) % report_frq == 0) or (iter == 0):    
            np.random.seed()
            score = test_svm(1000, dim, svm_classifier, accuracy_method)
            print('Current score is {} \n'.format(score))
            score_list.append(score)
            num_iter_list.append(iter)

        # if we want to see heatmap of uncertainty, we can activate the next line
        # plot_heatmap_uncertainty(Gaussian_regressor)

        # Optimize the objective function
        opt_x = [] # optimal X
        opt_fun = [] # optimal function value
        for i in range(n_optimization):
            np.random.seed()
            opt = minimize(fun, x0 = np.random.rand(dim), method = "L-BFGS-B", bounds=bounds)
            opt_x.append(opt.x)
            opt_fun.append(opt.fun)
        
        # Take the minimum value
        new_fun = min(opt_fun)
        
        # Find the corresponding X for the minimum value
        new_x = opt_x[np.argmin(opt_fun)]
        
        # Check whether there is a close point. If there is a close point, the sample is not added to the list
        if check_close_points(X, new_x):
            print('There is a similar point')
        
        else:
            # Add new_x to the training data
            X = np.vstack([X, new_x])
        
            if iter == 0 :
                new_points = np.atleast_2d(new_x)
            else:
                new_points = np.vstack([new_points, new_x])
            y.append(check_class(new_x, func=func, condition=condition))

            # Print
            np.set_printoptions(precision=3, suppress=True)
            print('Iteration {0} : Added point x value is {1} and function value is {2:2.2E}'.format(iter, new_x, new_fun))

        # Add iteration number
        iter += 1
    return score_list, num_iter_list, X 

#################################################################################################################
# Trainining with only sampling methods
def SamplingOnly_SVM(X_initial, max_itr, step_size, iteration, sampling_method, accuracy_method):
    ''' 
    Train SVM with data generated by sampling determined by method 
    These data are added to initial samples (X_initial) 

    X_initial : initial training data 

    max_itr : maximum number of samples

    step_size : report frequency

    iteration : number of iteration to calculate the mean/variance of svm accuracy score

    sampling_method: {'LHS' , 'Random'}

    accuracy_method: {'F1', 'MCC', 'Simple'}

    '''
    score_list = []
    dim = X_initial.shape[1]

    for _num_iter in np.arange(0, max_itr + step_size, step_size):
        _score_lst = []
        itr = 0
        while itr < iteration:
            if _num_iter == 0:
                X_final = X_initial

            else: 
                if sampling_method == 'LHS':
                    X_sample = lhs(dim, samples= _num_iter)
                elif sampling_method == 'Random':
                    X_sample = np.random.random([_num_iter, dim])
                else:
                    raise NotImplementedError('There is no such method')
                X_final = np.vstack([X_initial, X_sample])

            y_random = []
            for _X in X_final:
                y_random.append(check_class(_X, func=func, condition=condition))

            # Initial setting
            svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = svm_random_state)

            # Fit the data
            svm_classifier.fit(X_final,y_random)

            # Test
            score = test_svm(1000, dim, svm_classifier, accuracy_method)
            _score_lst.append(score)
            
            itr += 1
        score_list.append(_score_lst)
    return score_list



#################################################################################################################
'''
Functions for plotting
'''

def plot_progress_plot(num_iter_list, X_initial, opt_score_list, lhs_score_list, rand_score_list, title, method):
    '''
    Plot progress plot

    num_iter_list : array of number of samples

    X_initial : Initial samples

    opt_score_list : score list using optimization 

    lhs_score_list : score list using LHS

    rand_score_list : score list using Random sampling

    title : title of plot

    method : Method to calculate SVM accuracy {'F1', 'MCC', 'Simple'}
    '''
    # To calculate total number of samples 
    extended_num_iter_list = np.array(num_iter_list) + X_initial.shape[0]

    # Plot for the result using optimization
    plt.fill_between(extended_num_iter_list, np.max(opt_score_list, axis=0), np.min(opt_score_list, axis=0), alpha=0.3, color = 'g')
    plt.scatter(extended_num_iter_list, np.mean(opt_score_list, axis=0), color='g')
    plt.plot(extended_num_iter_list, np.mean(opt_score_list, axis=0), color='g', label='optimization')

    # Plot for the result using LHS
    plt.fill_between(extended_num_iter_list, np.max(lhs_score_list, axis=1), np.min(lhs_score_list, axis=1), alpha = 0.1, color='r')
    plt.scatter(extended_num_iter_list, np.mean(lhs_score_list, axis=1), color='r')
    plt.plot(extended_num_iter_list, np.mean(lhs_score_list, axis=1), color = 'r', label = 'LHS')

    # Plot for the result using random sampling
    plt.fill_between(extended_num_iter_list, np.max(rand_score_list, axis=1), np.min(rand_score_list, axis=1), alpha = 0.1, color='b')
    plt.scatter(extended_num_iter_list, np.mean(rand_score_list, axis=1), color='b')
    plt.plot(extended_num_iter_list, np.mean(rand_score_list, axis=1), color = 'b', label = 'Random')

    # Plot formatting
    plt.title(title)
    plt.xlabel('Samples')
    _ylabel = 'SVM accuracy (' + str(method) + ')'
    plt.ylabel(_ylabel)
    plt.legend()
    plt.show()
    return extended_num_iter_list


def plot_required_sample(threshold_start, threshold_stop, opt_score_list, lhs_score_list, rand_score_list, extended_num_itr_list):
    '''
    Plot desired accuracy vs required number of samples
    If mean score is above the threshold, then choose the number of samples to achieve that accuarcy
    '''
    threshold_accuracy = np.arange(threshold_start, threshold_stop, 0.01)
    mean_score_opt = np.mean(opt_score_list, axis=0)
    mean_score_lhs = np.mean(lhs_score_list, axis=1)
    mean_score_rand = np.mean(rand_score_list, axis=1)

    sample_opt = []
    sample_lhs = []
    sample_rand = []

    thr_valid_opt = []
    thr_valid_lhs = []
    thr_valid_rand = []

    for thr in threshold_accuracy:

        mean_score_opt_filtered = mean_score_opt[mean_score_opt < thr]
        opt_size = mean_score_opt_filtered.shape[0]
        mean_score_lhs_filtered = mean_score_lhs[mean_score_lhs < thr]
        lhs_size = mean_score_lhs_filtered.shape[0]
        mean_score_rand_filtered = mean_score_rand[mean_score_rand < thr]
        rand_size = mean_score_rand_filtered.shape[0]
        
        if (opt_size == 0 or opt_size == mean_score_opt.shape[0]):
            thr_valid_opt.append(False)
        else:
            itr_opt = max(extended_num_itr_list[mean_score_opt < thr])
            sample_opt.append(itr_opt)
            thr_valid_opt.append(True)

        if (lhs_size == 0 or lhs_size == mean_score_lhs.shape[0]):
            thr_valid_lhs.append(False)
        else:
            itr_lhs = max(extended_num_itr_list[mean_score_lhs < thr])
            sample_lhs.append(itr_lhs)
            thr_valid_lhs.append(True)

        if (rand_size == 0 or rand_size == mean_score_rand.shape[0]):
            thr_valid_rand.append(False)
        else:
            itr_rand = max(extended_num_itr_list[mean_score_rand < thr])
            sample_rand.append(itr_rand)
            thr_valid_rand.append(True)

#    minimum_plot_size = min(len(sample_opt), len(sample_lhs), len(sample_rand))

    plt.figure()        
    plt.plot(threshold_accuracy[thr_valid_opt], sample_opt, 'g-', label = 'Optimization')
    plt.plot(threshold_accuracy[thr_valid_lhs], sample_lhs, 'r--', label = 'LHS', alpha = 0.4)
    plt.plot(threshold_accuracy[thr_valid_rand], sample_rand, 'b--', label = 'Random', alpha = 0.4)

    title = 'Number of samples for desired accuracy (' + accuracy_method + ')'
    plt.title(title)
    plt.xlabel('Desired accuracy')
    plt.ylabel('Number of samples needed')
    plt.legend()
    plt.show()

##################################################################################################################################
'''
Functions for plotting only for 2-D problem
'''
def plot_heatmap_uncertainty(Gaussian_regressor):
    ''' 
    Plot heat map of uncertainty calculated by Gaussian regressor 
    Only for 2-D problem
    '''
    n_points = 10
    # Assume x1 and x2 are within [0,1]
    x1 = np.linspace(0,1,n_points)
    x2 = np.linspace(1,0,n_points)

    for i, _x2 in enumerate(x2):
        y_value = []
        for _x1 in x1:
            # Gaussian_regressor.predict can calculate uncertainty if return_std is True
            y_value.append(Gaussian_regressor.predict(np.atleast_2d([_x1,_x2]), return_std = True)[1][0])
        if i == 0:
            heatmap_data = np.array(y_value).reshape(1,n_points)
        else:
            heatmap_data = np.vstack([heatmap_data, np.array(y_value).reshape(1,n_points)])  
    sn.heatmap(heatmap_data)
    plt.show()

def plot_svm_boundary(svm_classifier, X, y):
    ''' 
    Plot svm decision boundary of svm_classifier with data X and y 
    Only for 2-D problem
    '''
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
    # If want to save the decision boundary plot
    #plt.savefig('./svm_classification_svmuncertainty_circle/final_boundary.png')
    plt.show()

def plot_scatter_data(svm_classifier, X, y, num_initial_sample):
    ''' 
    Scatter plot for data 
    Only for 2-D problem
    '''

    X_initial = X[:num_initial_sample, :]
    y_initial = y[:num_initial_sample]

    new_points = X[num_initial_sample:, :]
    
    # Initial samples
    plt.scatter(X_initial[:,0], X_initial[:,1], c=y_initial, s=30, alpha = 0.3)
    # New points
    plt.scatter(new_points[:,0], new_points[:,1], s=50, c = 'r', marker = '*')
    # Support vectors
    plt.scatter(svm_classifier.support_vectors_[:,0], 
                svm_classifier.support_vectors_[:,1], 
                s=15, marker='x')
    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.legend(['Initial points', 'new points', 'support vectors'])
    plt.show()


# %% Start of main loop
################################################################################################

'''
Generate X and y data
'''
svm_random_state = 42

# Benchmark Function
# Import function from Test_functions.py
func = Branin2d 
dim = 2 # number of features for benchmark function

# Set condition for feasible region 
condition = lambda x: func(x) <= -0.9

# Initial samples
num_samples = 5 # number of initial samples
X = initial_point_sampling(dim, num_samples, 'lhs')
y = []

for _X in X:
    y.append(check_class(_X, func=func, condition=condition))

# Check data feasibility
if 1 in y and -1 in y:
    print('Data contains both classifications. Good to go')    
else: 
    raise ValueError('One classification data is missing. Different random state for lhs is needed.')

# copy initial data
X_initial = X.copy()
y_initial = y.copy()
#%%
'''
Full loop start if initial data has all classifications (-1 and 1)
'''

max_itr = 300                # number of additional samples
C1 = 1e2                      # weight on the uncertainty
report_frq = 10               # report/test frequency
n_optimization = 3            # number of optimization in the main loop for one sample
max_main_loop = 3             # number of main loop to calculate mean/variance of the proposed algorithm
accuracy_method = 'F1'        # method to calculate svm accuracy {'F1', 'MCC', 'Simple'}

# Initial parameter setting
svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = svm_random_state) # initial svm
new_points = np.array([]) # initialize new points collection
# Variable bound
bounds = []
for i in range(dim):
    bounds.append((0.0, 1.0))   # variable bounds

# Loop start
opt_score_list = [] # initialize score list
for inner_itr in range(max_main_loop):
    y = y_initial.copy()
    score_list, num_iter_list, X_final = mainloop(svm_classifier = svm_classifier,
                                            num_iter = max_itr, bounds = bounds, n_optimization = n_optimization, 
                                            X = X, y=y, report_frq = report_frq, accuracy_method = accuracy_method)
    opt_score_list.append(score_list)

# %%
# LHS Sampling-based SVM
lhs_score_list = SamplingOnly_SVM(X_initial = X_initial, max_itr = max_itr, step_size = report_frq, iteration = 5, 
                                    sampling_method = 'LHS', accuracy_method = accuracy_method)

# %%
# Random sampling-based SVM
rand_score_list = SamplingOnly_SVM(X_initial = X_initial, max_itr = max_itr, step_size= report_frq, iteration = 5, 
                                    sampling_method = 'Random', accuracy_method= accuracy_method)

#%%
# Plot progress plot 
title = func.__name__ + ' with C1=' + str(C1) + ' Score = ' + accuracy_method
extended_num_itr_list = plot_progress_plot(num_iter_list= num_iter_list, X_initial = X_initial, opt_score_list=opt_score_list,
                    lhs_score_list = lhs_score_list, rand_score_list= rand_score_list,
                    title = title, method = accuracy_method)

# %%
# Plot required samples in terms of desired accuracy
plot_required_sample(0.1, 1.0, opt_score_list = opt_score_list, lhs_score_list=lhs_score_list, rand_score_list=rand_score_list, extended_num_itr_list = extended_num_itr_list)

#%%
# Save results
np.save("opt_score.npy", np.array(opt_score_list))
np.save("lhs_score.npy", np.array(lhs_score_list))
np.save("rand_score.npy", np.array(rand_score_list))
np.save("extended_num_itr_list.npy", np.array(extended_num_itr_list))
