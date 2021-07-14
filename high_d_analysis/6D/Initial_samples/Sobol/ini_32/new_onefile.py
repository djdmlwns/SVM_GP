
import numpy as np
from collections import Counter
from math import *
from statistics import mean
from pyDOE2 import lhs, ff2n, fullfact, bbdesign
# Trainining with only sampling methods
from pyDOE2 import lhs
from sklearn import svm
from contextlib import suppress
from scipy.optimize import minimize, NonlinearConstraint
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.utils.optimize import _check_optimize_result
import matplotlib.pyplot as plt
import seaborn as sn
import sobol

def check_class(x, case, **kwargs):
    ''' check classification of data x 
    #####################################
    Input: 
    
    case : {'benchmark', 'simulation'}

    **kwargs :{'condition'} should include feasibility constraint if case == benchmark

    ######################################
    
    '''
    # This is checked by function value for now
    # It will be determined by simulation for future use
    if case == 'benchmark': 
        if kwargs == None:
            raise ValueError('For benchmark case, function and feasibility condition should be set')
        else:
            condition = kwargs['condition']

    def run_simulation(x):
        sim = Simulation(x)
        sim.run()
        return sim.result

    def run_benchmark(x, condition):
        if condition(x): 
            return 1 # positive class (considered as feasible)
        else:
            return -1 # negative class (considered infeasible)        

    if case == 'benchmark':
        return run_benchmark(x, condition)

    elif case == 'simulation':
        return run_simulation(x)
    
    else: 
        raise NotImplementedError('Case should be either with benchmark function or simulation')


def test(num_test_points, dim, svm_classifier, check_class, case, num_itr_mean = 1, method = 'F1', **kwargs):
    ''' 
    Test prediction accuracy of SVM with 1000 random samples

    num_test_points : number of points for accuracy test

    dim : Number of features in X

    svm_classifier : svm classifier to test
    
    check_class : function to check class of test points

    case : {'benchmark' , 'simulation'}

    num_itr_mean : number of iteration to estimate the accuracy score (mean value)

    method: {'F1', 'MCC', 'Simple'}

        F1: F1-score

        MCC: Matthews correlation coefficient

        Simple: Simple accuracy (correct / total)

    **kwargs :{'condition'} should include feasibility constraint if case == benchmark  
    '''

    if case == 'benchmark': 
        if kwargs == None:
            raise ValueError('For benchmark case, function and feasibility condition should be set')
        else:
            condition = kwargs['condition']

    # Initialize score list
    score_lst = []

    # Start loop to calculate mean value of svm classification accuracy
    for itr in range(num_itr_mean):
        # Generate test points
        test_X = np.random.random([num_test_points, dim])

        # check true classification of data point and append to y
        test_y = []
        for _x in test_X:
            if case == 'benchmark':
                test_y.append(check_class(_x, case = case, condition = condition))
            else:
                test_y.append(check_class(_x, case = case))
                
        # get prediction of svm classifier
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
            
            # If no available method
            else:
                raise NotImplementedError('There is no such method for accuracy calculation')
        score_lst.append(score)

    return mean(score_lst)  # calculate mean value of accuracy

def check_data_feasible(y):
    # if there are both classes (-1 and 1 in y)
    if 1 in y and -1 in y:
        print('Data contains both classifications. Good to go')    
    
    # Raise error if there is only one
    else: 
        raise ValueError('One classification data is missing. More initial points are needed before start.')


###################################


class Initialization():
    '''
    Sampling object for initial data
    #####################################
    Input:
    
    dim : dimension of features

    num_samples : number of initial samples

    case : {'benchmark', 'simulation'}

    **kwargs : {'condition'} should include feasibility constraint for benchmark function

    #####################################

    '''
    def __init__(self, dim, num_samples, case = 'benchmark', **kwargs):
        self.dim = dim
        self.num_samples = num_samples      
        self.case = case

        if case == 'benchmark':
            if 'condition' in kwargs.keys():
                self.condition = kwargs['condition']
            else:
                raise ValueError('Benchmark function needs condition for feasibility')

    def sample(self, method = 'lhs'):
        '''
        Function to generate initial samples

        method: {'doe', 'ff', 'lhs', 'random', 'sobol', 'bb'}

            doe: Full factorial

            lhs: Latin Hypercube Sampling
            
            random: Random sampling

            sobol: use sobol sequence for quasi-montecarlo

            bb: box-behnken design (three levels for each factor)
            
        '''
        self.method = method
        y = []

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

        if method == 'doe':
            X = corner_addition(np.array([]), self.dim)
        elif method == 'ff':
            num_levels = int(self.num_samples**(1/self.dim))
            X = fullfact([num_levels for i in range(self.dim)]) / (num_levels - 1)
        elif method == 'lhs':
            # import lhs function from pyDOE2
            X = lhs(self.dim, self.num_samples)
        elif method == 'random':
            X = np.random.random([self.num_samples, self.dim])
        elif method == 'sobol':
            X = sobol.sample(dimension = self.dim, n_points = self.num_samples)
        elif method == 'bb':
            X = bbdesign(self.dim)
            X[X == 0.] = 0.5   # to change -1, 0, 1 to 0, 0.5, 1
            X[X == -1.] = 0.   # to change -1, 0, 1 to 0, 0.5, 1
        else:
            raise ValueError('No such method')

        for _X in X:
            if self.case == 'benchmark':
                y.append(check_class(_X, case = self.case, condition=self.condition))
            else:
                y.append(check_class(_X, case = self.case))

        return X, y

#####################################




class Sampling_based_SVM():
    def __init__(self, X_initial, max_itr, report_frq, iteration, sampling_method, 
                accuracy_method= 'F1', case = 'benchmark', svm_random_state = None, 
                **kwargs):
        ''' 
        Train SVM with data generated by sampling determined by method 
        These data are added to initial samples (X_initial) 

        #####################################################
        INPUT: 

        X_initial : initial training data 

        max_itr : maximum number of samples

        report_frq : report frequency for prediction accuracy score

        iteration : number of iterations to calculate the mean/variance of svm accuracy score

        sampling_method: {'lhs' , 'random', 'ff', 'bb', 'sobol'}

        accuracy_method: {'F1', 'MCC', 'Simple'}

        case: {'benchmark', 'simulation'}

        svm_random_state : random number for svm optimizer

        **kwargs: {condition: only for benchmark function, feasibility constraint should be given here} 

        #####################################################
        OUTPUT:

        self.score_list : svm accuracy score list

        '''
        self.X_initial = X_initial
        self.max_itr = max_itr
        self.report_frq = report_frq
        self.iteration = iteration
        self.sampling_method = sampling_method
        self.accuracy_method = accuracy_method
        self.svm_random_state = svm_random_state
        self.case = case

        self.score_list = []
        
        if case == 'benchmark': 
            if kwargs == None:
                raise ValueError('For benchmark case, function and feasibility condition should be set')
            else:                
                self.condition = kwargs['condition']

    def train(self):
        '''
        train svm using samples collected either randomly or using LHS
        '''
        X_initial = self.X_initial
        dim = X_initial.shape[1]
        report_frq = self.report_frq

        for _num_iter in np.arange(0, self.max_itr + report_frq, report_frq):
            _score_lst = []
            for itr in range(self.iteration):
                if _num_iter == 0:
                    X = X_initial.copy()

                else: 
                    # if LHS is used
                    if self.sampling_method == 'lhs':
                        X_sample = lhs(dim, samples= _num_iter)
                    # if random sampling is used
                    elif self.sampling_method == 'random':
                        X_sample = np.random.random([_num_iter, dim])
                    elif self.sampling_method == 'ff':
                        num_levels = int(_num_iter**(1/dim))
                        X_sample = fullfact([num_levels for i in range(dim)]) / (num_levels - 1)
                    elif self.sampling_method == 'sobol':
                        X_sample = sobol.sample(dimension = dim, n_points = _num_iter)
                    elif self.sampling_method == 'bb':
                        X = bbdesign(dim)
                        X[X == 0.] = 0.5   # to change -1, 0, 1 to 0, 0.5, 1
                        X[X == -1.] = 0.   # to change -1, 0, 1 to 0, 0.5, 1
                    # something else is specified
                    else:
                        raise NotImplementedError('There is no such method')


                    X = np.vstack([X_initial, X_sample])

                # check class of data points
                y = []
                for _X in X:
                    if self.case == 'benchmark':
                        y.append(check_class(_X, self.case, condition =  self.condition))
                    else:
                        y.append(check_class(_X, self.case))
                        
                # Initial setting
                svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = self.svm_random_state)

                # Fit the data
                svm_classifier.fit(X,y)

                # Test
                score = test(1000, dim, svm_classifier, check_class, case = self.case, method = self.accuracy_method, condition = self.condition)
                _score_lst.append(score)
                
            self.score_list.append(_score_lst)
        
###############################
# %%

class Simulation():
    '''
    This needs to be modified 
    '''
    def __init__(self, x, **kwargs):
        self.result = None
        self.x = x
        with open('./data/input.txt', 'w') as f:
            f.write(str(x))


    def run(self):
        # use data for simulation
#        data = self.x 
        self.send_data()
        
        result_text = self.retrieve_result()
        result = self.postprocessing_result(result_text)        
        self.result = result


    def send_data(self):
        # send text file to server for simulation
        # adequate formating should be known in the future
        print('Data is sent')
        pass


    def retrieve_result(self):
        # read result text file from server
        with open('./data/output.txt', 'r') as f:
            lines = f.read()
        return lines


    def postprocessing_result(self, result_text):
        # postprocess text to be used in Python
        processed_result = result_text
        return processed_result 

####################################################  
#       #
# %%



def HARTMANN6D(x):
    ''' x needs to be array '''
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    
    A = np.array([[10., 3., 17., 3.5, 1.7, 8],
                [0.05, 10., 17., 0.1, 8., 14.],
                [3., 3.5, 1.7, 10., 17., 8.],
                [17., 8., 0.05, 10., 0.1, 14.],]
                )

    P = 1e-4 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                        [2329., 4135., 8307., 3736., 1004., 9991.],
                        [2348., 1451., 3522., 2883., 3047., 6650.],
                        [4047., 8828., 8732., 5743., 1091., 381.]])
    
    fun = - sum(alpha[i] * exp(-sum(A[i,j] * (x[j] - P[i,j])**2 for j in range(6))) for i in range(4))

    return fun 

def HARTMANN4D(x):
    ''' x needs to be array 
    mean of zero and variance of one
    '''
    alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
    
    A = np.array([[10., 3., 17., 3.5, 1.7, 8],
                [0.05, 10., 17., 0.1, 8., 14.],
                [3., 3.5, 1.7, 10., 17., 8.],
                [17., 8., 0.05, 10., 0.1, 14.],]
                )

    P = 1e-4 * np.array([[1312., 1696., 5569., 124., 8283., 5886.],
                        [2329., 4135., 8307., 3736., 1004., 9991.],
                        [2348., 1451., 3522., 2883., 3047., 6650.],
                        [4047., 8828., 8732., 5743., 1091., 381.]])
    
    fun = 1/0.839 * (1.1 - sum(alpha[i] * exp(-sum(A[i,j] * (x[j] - P[i,j])**2 for j in range(4))) for i in range(4)))

    return fun 

def Dette8d(x):
    fun = 4 * (x[0] - 2 + 8 * x[1] - 8 *x[1] ** 2 )**2 + (3 - 4 * x[1])**2 \
           + 16*sqrt(x[2] +1) * (2*x[2] - 1) **2 + sum(log(1+sum(x[j] for j in range(2,i))) for i in range(4,8)) 
    return fun        


def Rosenbrock4d(x):
    x_bar = 15 * x - 5
    fun = 1/(3.755 * 10**5) *(sum(100 * (x_bar[i] - x_bar[i+1]**2)**2 + (1-x_bar[i])**2 for i in range(2)) - 3.827 * 10 **5)
    return fun


def Branin2d(x):
    x1 = 15 * x[0] - 5
    x2 = 15 * x[1] 
    fun =  1/51.95 * ((x2 - 5.1 * x1**2 / (4 * pi **2) + 5 * x1 / pi - 6)**2 \
            + (10 - 10 / (8 * pi)) * cos(x1) - 44.81)
    return fun

def Hosaki2d(x):
    x1 = 10 * x[0]
    x2 = 10 * x[1]
    fun = (1 - 8 * x1 + 7 * x1 **2 - 7/3 * x1**3 + 1/4 * x1 **4 ) * x2 **2 * exp(-x2)
    return fun
# %%


##################################
# %%

# %%
#############################################################################################################
class ActiveSampling():
    def __init__(self, X_initial, y_initial, SVM_classifier, GP_regressor, bounds, 
                max_itr = 300, verbose = True, C1 = 100, p_check = 0.1, threshold = 1, n_optimization = 5, case = 'benchmark', 
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
#    @profile
    def train(self):
        '''
        Train SVM and GP to choose next optimal sample, and repeat training until the maximum iteration
        '''
        with open('log.txt', 'a') as f:
            for iter in range(self.max_itr):
                self._iter = iter
                # Fit svm
                self.SVM_classifier.fit(self.X, self.y)

                # Define kernel for GP (Constant * RBF)
                # Note that length scale is changed when data X is changed
                kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-5, 1e5)) \
                        * RBF(length_scale=sqrt(self.X.var() * self.X.shape[1]/2), length_scale_bounds='fixed') 

                # Define GP
                self.GP_regressor.kernel = kernel

                del kernel

                # Calculate g(x) using svm
                continuous_y = self.SVM_classifier.decision_function(np.atleast_2d(self.X))

		        # Train gaussian process regressor using X and continous y
                self.GP_regressor.fit(self.X, continuous_y) 

                del continuous_y

                self.C_lst.append(self.GP_regressor.kernel_.get_params()['k1__constant_value'])
                # Find the next sampling point
                new_x, new_fun = self.optimize_acquisition()

                # Check whether there is a close point. If there is a close point, the sample is not added to the list
                # Run up-to five times to find different points
                for _itr in range(5): 
                    if self.check_close_points(new_x):
                        if self.verbose:
#                            print('There is a similar point')
#                            print('Iteration {0} : point x value is {1} but not added'.format(iter, new_x))
                            pass
                            
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
#                                print('Iteration {0} : Added point x value is {1} and function value is {2:2.2E}\n'.format(iter, new_x, new_fun), file=f)
                                pass
                            else:
#                                print('Iteration {0} : Added point x value is {1} and function value is {2:2.2E}\n'.format(iter, new_x, new_fun))
                                pass
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
#                                print('Current score is {} \n'.format(score), file=f)
                                pass
                            else:
#                                print('Current score is {} \n'.format(score))
                                pass
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

                del opt
        
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

                del opt
        
        # Take the minimum value
        new_fun = min(opt_fun)
        
        # Find the corresponding X for the minimum value
        new_x = opt_x[np.argmin(opt_fun)]

        del opt_fun
        del opt_x 

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


####################################################

#################################################################################################################

#################################################################################################################
'''
Functions for plotting
'''


def progress_plot(num_iter_list, X_initial, opt_score_list, lhs_score_list, rand_score_list, title, method, path = None):
    '''
    Plot progress plot
    ############################################
    INPUT:

    num_iter_list : array of number of samples

    X_initial : Initial samples

    opt_score_list : score list using active sampling  

    lhs_score_list : score list using LHS

    rand_score_list : score list using Random sampling

    title : title of plot

    method : Method to calculate SVM accuracy {'F1', 'MCC', 'Simple'}
    '''
    # To calculate total number of samples 
    extended_num_iter_list = np.array(num_iter_list) + X_initial.shape[0]

    # Plot for the result using optimization
    plt.fill_between(extended_num_iter_list, np.max(opt_score_list, axis=1), np.min(opt_score_list, axis=1), alpha=0.3, color = 'g')
    plt.scatter(extended_num_iter_list, np.mean(opt_score_list, axis=1), color='g')
    plt.plot(extended_num_iter_list, np.mean(opt_score_list, axis=1), color='g', label='optimization')

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
    
    if path != None:
        plt.savefig(path + '/pgp_' + title + '.png')

    plt.show()


def required_sample(threshold_start, threshold_stop, opt_score_list, lhs_score_list, rand_score_list, num_iter_list, X_initial, 
                    accuracy_method, title = None, path=None):
    '''
    Plot desired accuracy vs required number of samples
    If mean score is above the threshold, then choose the number of samples to achieve that accuarcy
    ##################################################################
    INPUT:
    
    threshold_start : start value for desired svm accuracy

    threshold_stop : final value for desired svm accuracy

    opt_score_list : score list using active sampling  

    lhs_score_list : score list using LHS

    rand_score_list : score list using Random sampling

    num_iter_list: number of iteration list

    X_initial : initial training data

    accuracy method : Method to calculate SVM accuracy {'F1', 'MCC', 'Simple'}

    '''
    # To calculate total number of samples 
    extended_num_itr_list = np.array(num_iter_list) + X_initial.shape[0]
    
    threshold_accuracy = np.arange(threshold_start, threshold_stop, 0.01)
    mean_score_opt = np.mean(opt_score_list, axis=1)
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

    if title == None:
        title = 'Number of samples for desired accuracy (' + accuracy_method + ')'

    plt.title(title)
    plt.xlabel('Desired accuracy')
    plt.ylabel('Number of samples needed')
    plt.legend()
    
    if path != None:
        plt.savefig(path + '/sample_' + title + '.png')
    
    plt.show()



##################################################################################################################################
# Functions for plotting only for 2-D problem

def plot_heatmap_uncertainty(Gaussian_regressor):
    ''' 
    Plot heat map of uncertainty calculated by Gaussian regressor 
    Only for 2-D problem
    '''
    n_points = 30
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


def plot_svm_boundary(svm_classifier, X, y, **kwargs):
    ''' 
    Plot svm decision boundary of svm_classifier with data X and y 
    Only for 2-D problem
    '''
    xx, yy = np.meshgrid(np.linspace(0, 1, 500),
                        np.linspace(0, 1, 500))

    # plot the decision function for each datapoint on the grid
    Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='none',
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
    if 'path' in kwargs.keys():
        path = kwargs['path'] + '/final_decisionboundary.png'
        plt.savefig(path)
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

##########################################################
# %%

# %%
# Benchmark Function Test Main script
case = 'benchmark' # case can be 'benchmark' or 'simulation'
dim = 6 # number of features for benchmark function
sample_method = 'sobol'
# Set condition for feasible region  
# This is only needed for benchmark function (case = 'benchmark')
condition = lambda x: HARTMANN6D(x) <= -0.5

# Initial samples
num_samples = 2**5    # number of initial samples
Initializer = Initialization(dim, num_samples, case = case, condition = condition) # Class for initial sampling

# Check data feasibility
# Raise ValueError if only one classification (e.g., only 1 and no -1) is included in the initial sample
# Sampling again if the error is raised
for itr in range(10):
    try: 
        X, y = Initializer.sample(sample_method)
        check_data_feasible(y) 
        break
    except ValueError:
        print('Need to resample') 

#%%
max_main_loop = 3             # number of main loop to calculate mean/variance of the svm accuracy of the proposed algorithm -> need for plot
accuracy_method = 'F1'        # method to calculate svm accuracy {'F1', 'MCC', 'Simple'} are available. Default is F1-score
max_itr = 500                # maximum number of samples
report_frq = 10               # frequency to test svm accuracy and print

# Variable bound
bounds = []
for i in range(dim):
    bounds.append((0.0, 1.0))   # [0, 1]^n. For simulation, normalization of X is needed

# Start Active Learning/Sampling algorithm 
opt_score_list = [] # initialize score list 

for _itr in range(max_main_loop):
    # initialize SVM
    SVM_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42) 
    
    # initialize GP regressor
    GP_regressor = MyGPR(normalize_y = True, n_restarts_optimizer = 0, alpha = 1e-7, copy_X_train = False) 
    
    # initialize Active sampling algorithm class 
    AS = ActiveSampling(X, y, SVM_classifier, GP_regressor, bounds, 
                        max_itr = max_itr, case = case, C1=1, accuracy_method = accuracy_method, n_optimization =  5, report_frq = report_frq,
                        condition = condition, p_check = 0.0, threshold = 1, C1_schedule='None', acq_type = 'f1')
    # training start
    AS.train()

    # append accuracy score to the list
    opt_score_list.append(AS.score_list)

    del SVM_classifier
    del GP_regressor
# change the shape of list for future plotting
opt_score_list = (np.array(opt_score_list).T).tolist()

# %%
# SVM trained without using Active sampling algorithm
# 1) LHS Sampling-based SVM

# Initialize class instance
# sampling_method = 'lhs'
SS_LHS = Sampling_based_SVM(X, max_itr = max_itr,
                        report_frq = report_frq, iteration = 3, sampling_method = 'lhs', accuracy_method = accuracy_method, 
                        case = case, svm_random_state = 42, condition = condition)

# train
SS_LHS.train()

# save the svm accuracy score
lhs_score_list = SS_LHS.score_list

# 2) Random Sampling-based SVM
# Initialize class instance
# sampling_method = 'random'
SS_Rand = Sampling_based_SVM(X, max_itr = max_itr, 
                        report_frq = report_frq, iteration = 3, sampling_method = 'random', accuracy_method = accuracy_method, 
                        case = case, svm_random_state = 42, condition = condition)

# train
SS_Rand.train()

# save the svm accuracy score
rand_score_list = SS_Rand.score_list

#########################################################################################
# Plot section
#########################################################################################
#%%
# Plot progress plot 
# set title (e.g., C1=100 / Score = F1)
title = 'SampleMethod_' + sample_method + '_Initial_' + str(AS.X_initial.shape[0]) + '_C1_' + str(AS.C1) + '_Score_' + accuracy_method + '_AC_' + AS.acq_type + '_C1_schedule_' + str(AS.C1_schedule)# set title

# Plot svm accuracy improvement w.r.t. number of samples
# Require three score lists of SVM trained by AS/LHS/RANDOM
progress_plot(num_iter_list= AS.num_iter_list, X_initial = AS.X_initial, opt_score_list=opt_score_list,
                    lhs_score_list = lhs_score_list, rand_score_list= rand_score_list,
                    title = title, method = accuracy_method, path = '.')


# Plot required sample numbers in terms of desired accuracy
required_sample(0.1, 1.0, opt_score_list = opt_score_list, lhs_score_list=lhs_score_list, rand_score_list=rand_score_list, num_iter_list = AS.num_iter_list,
                        X_initial = X, accuracy_method = accuracy_method, title = title, path = '.')

#%%

# Plot heatmap of uncertainty to check only for 2-D
if dim == 2:
    plot_heatmap_uncertainty(AS.GP_regressor)

# %%
# Plot SVM boundary of the proposed algorithm if needed only for 2-D
if dim == 2:
    plot_svm_boundary(AS.SVM_classifier, AS.X, AS.y)
