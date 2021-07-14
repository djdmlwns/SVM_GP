# Sampling methods
from pyDOE2 import lhs, ff2n
import numpy as np
from Auxiliarty_function_svm import check_class
from Simulation import Simulation

class Initialization():
    def __init__(self, dim, num_samples, case = 'benchmark', **kwargs):
        self.dim = dim
        self.num_samples = num_samples      
        self.case = case
        if 'condition' in kwargs.keys():
            self.condition = kwargs['condition']

    def sample(self, method = 'lhs'):
        '''
        Function to generate initial samples
        method: {'doe', 'lhs', 'random'}
        doe: Full factorial
        lhs: Latin Hypercube Sampling
        random: Random sampling
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
        elif method == 'lhs':
            # import lhs function from pyDOE2
            X = lhs(self.dim, self.num_samples)
        elif method == 'random':
            X = np.random.random([self.num_samples, self.dim])
        else:
            raise ValueError('No such method')

        for _X in X:
            if self.case == 'benchmark':
                y.append(check_class(_X, case = self.case, condition=self.condition))
            else:
                y.append(check_class(_X, case = self.case))

        return X, y
    # def check_class(self, x):
    #     ''' check classification of data x '''
    #     # This is checked by function value for now
    #     # It will be determined by simulation for future use
    #     def run_simulation(x):
    #         sim = Simulation()
    #         sim.run(x)
    #         return sim.result

    #     def run_benchmark(x, condition):
    #         if condition(x): 
    #             return 1 # positive class (considered as feasible)
    #         else:
    #             return -1 # negative class (considered infeasible)        

    #     if self.case == 'benchmark':
    #         return run_benchmark(x, self.condition)

    #     elif self.case == 'simulation':
    #         return run_simulation(x)
        
    #     else: 
    #         raise NotImplementedError('Case should be either with benchmark function or simulation')