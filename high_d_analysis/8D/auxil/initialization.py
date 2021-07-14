# Sampling methods
from numpy.core.numeric import full
from pyDOE2 import lhs, ff2n, bbdesign, fullfact
import numpy as np
from auxil.auxilfunc import check_class
import sobol

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
