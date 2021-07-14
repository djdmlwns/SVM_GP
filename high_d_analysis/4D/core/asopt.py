# %%
from contextlib import suppress
from scipy.optimize import minimize, NonlinearConstraint
from math import sqrt, log
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

import numpy as np

from sklearn.utils.optimize import _check_optimize_result
from auxil.auxilfunc import test, check_class

# %%
#############################################################################################################
class ActiveSampling():
    def __init__(self, X_initial, y_initial, SVM_classifier, GP_regressor, bounds, 
                max_itr = 300, verbose = True, C1 = 100, p_check = 0.1, threshold = 1, n_optimization = 10, case = 'benchmark', 
                report_frq = 5, accuracy_method = 'F1', C1_schedule = None, acq_type = 'f1', log = False, **kwargs):

        '''
        Input: 
        ################################################################
        X_initial, y_initial: initial data {X: array, y: list/array}
        
        SVM_classifier: initial svm classifier

        GP_regressor: initial GP regressor

        bounds: variable bounds 

        max_itr : maximum number of additional sampling

        verbose : If false, no print output

        C1 : weight on uncertainty

        p_check : probability to solve constrained optimization to check inside feasible region

        threshold : threshold on g(x) (g(x) > threshold) when solving constrained problem

        n_optimization : number of re-initialization for acquisition function optimization

        case : {'benchmark', 'simulation'}

        report_frq : frequency to test svm accuracy

        accuracy_method : Method to measrue svm classification accuracy {'F1', 'MCC', 'Simple'}

        **kwargs : {'condition'} needs to include the feasibility condition if case == benchmark

        Output: 
        ###################################################################
        self.X / self.y : Final data

        self.score_list : svm accuracy score list

        self.new_points : Samples selected by Active learning (except initial points)

        '''
        self.X_initial = X_initial.copy() 
        self.y_initial = y_initial.copy() 
        self.SVM_classifier = SVM_classifier
        self.GP_regressor = GP_regressor
        self.bounds = bounds
        self.max_itr = max_itr
        self.verbose = verbose
        self.C1_initial = C1
        self.C1 = C1
        self.p_check = p_check
        self.threshold = threshold
        self.n_optimization = n_optimization
        self.case = case
        self.report_frq = report_frq
        self.accuracy_method = accuracy_method
        self.C1_schedule = C1_schedule
        self.acq_type = acq_type
        self.log = log

        self.X = X_initial
        self.y = y_initial.copy()
        self.score_list = []
        self.num_iter_list = []
        self.new_points = np.array([], dtype = np.float64)
        self.dim = X_initial.shape[1]
        self.C_lst = []


        if case == 'benchmark': 
            if kwargs == None:
                raise ValueError('For benchmark case, function and feasibility condition should be set')
            else:
                self.condition = kwargs['condition']

    def train(self):
        '''
        Train SVM and GP to choose next optimal sample, and repeat training until the maximum iteration
        '''
        with open('log.txt', 'a') as f:
            for iter in range(self.max_itr):
                self._iter = iter
                # Fit svm
                self.SVM_classifier.fit(self.X, self.y)

                # Calculate g(x) using svm
                continuous_y = self.SVM_classifier.decision_function(np.atleast_2d(self.X))

                # Define kernel for GP (Constant * RBF)
                # Note that length scale is changed when data X is changed
                kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e5)) \
                        * RBF(length_scale=sqrt(self.X.var() * self.X.shape[1]/2), length_scale_bounds='fixed') 

                # Define GP
                self.GP_regressor.kernel = kernel

                # Train gaussian process regressor using X and continous y
                self.GP_regressor.fit(self.X, continuous_y) 
                
                self.C_lst.append(self.GP_regressor.kernel_.get_params()['k1__constant_value'])
                # Find the next sampling point
                new_x, new_fun = self.optimize_acquisition()

                # Check whether there is a close point. If there is a close point, the sample is not added to the list
                # Run up-to five times to find different points
                for _itr in range(5): 
                    if self.check_close_points(new_x):
                        if self.verbose:
                            print('There is a similar point')
                            print('Iteration {0} : point x value is {1} but not added'.format(iter, new_x))
                            
                        # Resample
                        new_x, new_fun = self.optimize_acquisition()     

                        # back to loop
                        continue   

                    else:
                        # Add new_x to the training data
                        self.X = np.vstack([self.X, new_x])
                    
                        # If new_point is empty
                        if self.new_points.shape[0] == 0 :
                            self.new_points = np.atleast_2d(new_x)
                        # If new_point is not empty, stack the new point
                        else:
                            self.new_points = np.vstack([self.new_points, new_x])

                        # check classification of new_x and append to y
                        if self.case == 'benchmark':
                            # check with benchmark function value
                            self.y.append(check_class(new_x, self.case, condition = self.condition))
                        else:
                            # check with simulation
                            self.y.append(check_class(new_x, self.case))

                        # Print
                        if self.verbose:
                            np.set_printoptions(precision=3, suppress=True)
                            if self.log:
                                print('Iteration {0} : Added point x value is {1} and function value is {2:2.2E}\n'.format(iter, new_x, new_fun), file=f)
                            else:
                                print('Iteration {0} : Added point x value is {1} and function value is {2:2.2E}\n'.format(iter, new_x, new_fun))
                        break

                # Test svm and append score and iteration number to list
                # Only possible for benchmark function (Not possible for simulation)
                if self.case == 'benchmark':
                    if ((iter+1) % self.report_frq == 0) or (iter == 0):    
                        np.random.seed()
                        # Test svm accuarcy 
                        score = test(1000, self.dim, self.SVM_classifier, check_class, self.case, method = self.accuracy_method, condition = self.condition)
                        
                        if self.verbose:
                            if self.log: 
                                print('Current score is {} \n'.format(score), file=f)
                            else:
                                print('Current score is {} \n'.format(score))
                        
                        self.score_list.append(score)
                        self.num_iter_list.append(iter)

        return None 

    def acquisition_function(self, x):
        ''' Objective function to be minimized '''            
        # g(x) : 0 at the decision boundary
        fun_val = (self.value_prediction_svm(x))[0] 
        
        # U(x) : Uncertainty estimation
        uncertainty = self.GP_regressor.predict(np.atleast_2d(x), return_std = True)[1][0] 

        if self.C1_schedule == 'linear':
            self.C1 = self.C1_initial - (self.C1_initial - 1) * (self._iter) / (self.max_itr - 1)

        if self.acq_type == 'f1':
            return abs(fun_val) - self.C1 * (uncertainty)

        elif self.acq_type == 'f2':
            return fun_val**2 - self.C1 * log(uncertainty)

        elif self.acq_type == 'f3':
            return fun_val**2 - self.C1 * (uncertainty)

        else:
            raise ValueError('No such objective function form')
        

    def optimize_acquisition(self):
        '''optimize acquisition function'''
        opt_x = [] # optimal X list
        opt_fun = [] # optimal function value list
        
        # if random number is less than 1 - p_check, we solve unconstrained problem 
        # In the unconstrained problem, g(x) and U(x) are determined using trade-off
        if np.random.random() < 1 - self.p_check: 
            for _i in range(self.n_optimization):
                np.random.seed()
                # solve unconstrained problem
                opt = minimize(self.acquisition_function, x0 = np.random.rand(self.dim), method = "L-BFGS-B", bounds=self.bounds)
                
                opt_x.append(opt.x)
                opt_fun.append(opt.fun)
        
        # if random number is greater than 1-p_check, we solve "constrained" problem
        # min (-U(x)) s.t. g(x) > self.threshold
        # this is to check the points inside our feasible region determined by SVM machine
        else: 
            for _i in range(self.n_optimization):
                # constraint = NonlinearConstraint(lambda x : self.value_prediction_svm(x)[0], lb = self.threshold, ub = np.inf, jac = '2-point')
                # obj = lambda x : -self.GP_regressor.predict(np.atleast_2d(x), return_std= True)[1][0]
                # opt = minimize(obj, x0 = np.random.rand(self.dim), constraints = constraint, method = "SLSQP", bounds = self.bounds)
                obj = lambda x : - (self.value_prediction_svm(x))[0] - self.GP_regressor.predict(np.atleast_2d(x), return_std= True)[1][0]
                opt = minimize(obj, x0 = np.random.rand(self.dim), method = "L-BFGS-B", bounds=self.bounds)

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


# %%
class MyGPR(GaussianProcessRegressor):
    ''' 
    To change solver options of GP regressor (e.g., maximum number of iteration of nonlinear solver) 
    '''
    def __init__(self, *args, max_iter=3e6, max_fun = 3e6, **kwargs):
        super().__init__(*args, **kwargs)
        # To change maximum iteration number
        self._max_iter = max_iter
        self._max_fun = max_fun

    # _constrained_optimization is the function for optimization inside GaussianProcessRegressor
    # Redefine this to change the default setting for the maximum iteration
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            # change maxiter option
            opt_res = minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter, 'maxfun': self._max_fun })
            # _check_optimize_result is imported from sklearn.utils.optimize 
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
