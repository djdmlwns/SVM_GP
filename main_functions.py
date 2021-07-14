# %%
from contextlib import suppress
from sklearn import svm
from scipy.optimize import minimize
from copy import deepcopy
from pyDOE2 import *
from statistics import * 
from math import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

from sklearn.utils.optimize import _check_optimize_result
from collections import Counter
from Test_functions import *
from Auxiliarty_function_svm import test, check_class
from Simulation import Simulation
from Initialization import Initialization

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
class ActiveSampling():
    def __init__(self, X_initial, y_initial, SVM_classifier, GP_regressor, bounds, max_itr = 300, verbose = True, C1 = 100, n_optimization = 10, case = 'benchmark', 
                report_frq = 5, accuracy_method = 'F1', *args, **kwargs):
        self.X_initial = X_initial.copy()
        self.y_initial = y_initial.copy()
        self.X = X_initial
        self.y = y_initial.copy()
        self.SVM_classifier = SVM_classifier
        self.GP_regressor = GP_regressor
        self.bounds = bounds
        self.max_itr = max_itr
        self.verbose = verbose
        self.C1 = 100
        self.n_optimization = 10
        self.score_list = []
        self.num_iter_list = []
        self.new_points = np.array([], dtype = np.float64)
        self.case = case
        self.report_frq = report_frq
        self.accuracy_method = accuracy_method

        self.dim = X_initial.shape[1]

        if case == 'benchmark': 
            if kwargs == None:
                raise ValueError('For benchmark case, function and feasibility condition should be set')
            else:
                self.condition = kwargs['condition']

    def train(self):
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
        X_loop = self.X_initial

        for iter in range(self.max_itr):
            # Fit svm
            self.SVM_classifier.fit(X_loop, self.y)

            # Calculate g(x) using svm
            continuous_y = self.SVM_classifier.decision_function(np.atleast_2d(X_loop))

            # Define kernel for GP (Constant * RBF)
            kernel = ConstantKernel(constant_value=1.0, constant_value_bounds='fixed') \
                    * RBF(length_scale=sqrt(X_loop.var() * X_loop.shape[1]/2), length_scale_bounds='fixed') 
                    #  \
                    #  + ConstantKernel(constant_value = 10.0, constant_value_bounds = (1e-5, 1e5)) \
                    #  * RBF(length_scale = 1, length_scale_bounds=(sqrt(X.var() * X.shape[1]/2), 1e5))
                    # Length scale is determined as same as that of SVM RBF kernel

            # Define GP
            self.GP_regressor.kernel = kernel

            # Gaussian_regressor = MyGPR(kernel = kernel, normalize_y = True, n_restarts_optimizer = 0, alpha = 1e-7) 
            # Train gaussian process regressor using X and continous y
            self.GP_regressor.fit(X_loop, continuous_y) 
    #        print('Length scale is {} \n'.format(Gaussian_regressor.kernel_.get_params()['k2__length_scale']))
            # if we want to see heatmap of uncertainty, we can activate the next line
            # plot_heatmap_uncertainty(Gaussian_regressor)

            # Find the next sampling point

            new_x, new_fun = self.optimize_acquisition()

            # Check whether there is a close point. If there is a close point, the sample is not added to the list
            if self.check_close_points(new_x):
                if self.verbose:
                    print('There is a similar point\n')
                    print('Iteration {0} : point x value is {1} but not added'.format(iter, new_x))

            else:
                # Add new_x to the training data
                X_loop = np.vstack([X_loop, new_x])
            
                if self.new_points.shape[0] == 0 :
                    self.new_points = np.atleast_2d(new_x)
                else:
                    self.new_points = np.vstack([self.new_points, new_x])

                # add classification to y
                if self.case == 'benchmark':
                    self.y.append(check_class(new_x, self.case, condition = self.condition))
                else:
                    self.y.append(check_class(new_x, self.case))

                # Print
                np.set_printoptions(precision=3, suppress=True)
                print('Iteration {0} : Added point x value is {1} and function value is {2:2.2E}'.format(iter, new_x, new_fun))

            # Test svm and append score and iteration number to list

            if self.case == 'benchmark':
                if ((iter+1) % self.report_frq == 0) or (iter == 0):    
                    np.random.seed()
                    score = test(1000, self.dim, self.SVM_classifier, check_class, 'benchmark', method = self.accuracy_method, condition = self.condition)
                    
                    if self.verbose:
                        print('Current score is {} \n'.format(score))
                    
                    self.score_list.append(score)
                    self.num_iter_list.append(iter)

        self.X = X_loop

        return None 

    def acquisition_function(self, x):
        ''' Objective function to be minimized '''            
        # g(x) 
        fun_val = abs((self.value_prediction_svm(x))[0]) 
        
        # U(x) 
        uncertainty = self.GP_regressor.predict(np.atleast_2d(x), return_std = True)[1][0] 

        return fun_val - self.C1 * uncertainty


    def optimize_acquisition(self):
        # Optimize the objective function
        opt_x = [] # optimal X
        opt_fun = [] # optimal function value

        for i in range(self.n_optimization):
            np.random.seed()
            opt = minimize(self.acquisition_function, x0 = np.random.rand(dim), method = "L-BFGS-B", bounds=self.bounds)
            opt_x.append(opt.x)
            opt_fun.append(opt.fun)
        
        # Take the minimum value
        new_fun = min(opt_fun)
        
        # Find the corresponding X for the minimum value
        new_x = opt_x[np.argmin(opt_fun)]
                    
        return new_x, new_fun


    def value_prediction_svm(self, x):    
        ''' calculate g(x) value of point '''
        value_prediction = self.SVM_classifier.decision_function(np.atleast_2d(x))   
        return value_prediction # g(x) value    


    def check_close_points(self, x):
        ''' To check whether there are close data around the new sample point '''
        distance = self.X - x
        norm_set = np.linalg.norm(distance, axis = 1) # 2-norm of distances

        if np.any(norm_set < 1e-8):
            return True 
        else:
            return False   
