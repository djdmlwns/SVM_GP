
import numpy as np
from collections import Counter
from math import sqrt
from statistics import mean
from auxil.simulation import Simulation

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

