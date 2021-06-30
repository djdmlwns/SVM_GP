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


def test_svm(num_test_points, svm_classifier, initial_classifier, dim, check_class):
    ''' test final svm accuracy and compare with initial svm '''
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