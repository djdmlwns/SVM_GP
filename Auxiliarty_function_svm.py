import numpy as np
from collections import Counter
from math import sqrt
from statistics import mean
from Simulation import Simulation

def check_class(x, case, **kwargs):
    ''' check classification of data x '''
    # This is checked by function value for now
    # It will be determined by simulation for future use
    if case == 'benchmark': 
        if kwargs == None:
            raise ValueError('For benchmark case, function and feasibility condition should be set')
        else:
            condition = kwargs['condition']

    def run_simulation(x):
        sim = Simulation()
        sim.run(x)
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
    Test prediction accuracy of SVM 

    num_test_points : number of points for accuracy test

    dim : Number of features in X

    check_class : function to check class of test points

    method: {'F1', 'MCC', 'Simple'}

    F1: F1-score

    MCC: Matthews correlation coefficient

    Simple: Simple accuracy (correct / total)
    '''

    if case == 'benchmark': 
        if kwargs == None:
            raise ValueError('For benchmark case, function and feasibility condition should be set')
        else:
            condition = kwargs['condition']

    # Maximum iteration for mean value
    max_itr = num_itr_mean
    score_lst = []
    for itr in range(max_itr):
        test_X = np.random.random([num_test_points, dim])
        test_y = []
        for _x in test_X:
            if case == 'benchmark':
                test_y.append(check_class(_x, case = case, condition = condition))
            else:
                test_y.append(check_class(_x, case = case))
                
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

    return mean(score_lst) 


