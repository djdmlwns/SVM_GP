# %%
from sklearn import svm
from scipy.optimize import minimize
from copy import deepcopy
from pyDOE2 import *
from statistics import * 
from math import *
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

# %%
''' Functions definition '''
def value_prediction_svm(svm_classifier, point):    
    ''' calculate g(x) of point '''
    value_prediction = svm_classifier.decision_function(np.atleast_2d(point))   
    return value_prediction

def svm_uncertainty(svm_classifier, training_data, test_point):
    ''' calculate uncertainty of test_point '''
    if svm_classifier.get_params()['gamma'] == 'auto':
        gamma = 1/svm_classifier.shape_fit_[1]
        similarity = np.exp(-np.linalg.norm(training_data - test_point, axis = 1) * gamma).sum()
    return similarity

def check_class(x):
    ''' check classification of data x '''
    if (x[0] - 0.5)**2 +  (x[1] - 0.5)**2 <= 0.2 **2:
#    if (x[0] <= x[1] and x[1] >= -x[0] +1) or (x[0] >= x[1] and x[1] <= -x[0] +1):
        return 1
    else:
        return -1

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

def plot_scatter_data(svm_classifier, X,y, new_points = None):
    ''' scatter plot for data '''
    if type(new_points) == "NoneType":
        plt.scatter(X[:initial_sample_number,0], X[:initial_sample_number,1], c=y[:initial_sample_number], s=30, alpha = 0.3)
        plt.scatter(new_points[:,0], new_points[:,1], s=50, c = 'r', marker = '*')
    #plt.plot(x,y_hyperplane)
    else:
        plt.scatter(X[:,0], X[:,1], c=y, s=30, alpha = 0.3)
#    plt.scatter(svm_classifier.support_vectors_[:,0], 
#                svm_classifier.support_vectors_[:,1], 
#                s=15, marker='x')

    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.legend(['Initial points', 'new points'])
    plt.show()

def test_svm(num_test_points, svm_classifier, initial_classifier):
    ''' test final svm accuracy and compare with initial svm '''
    test_X = np.array([np.random.random(num_test_points), np.random.random(num_test_points)]).T
    test_y = []
    for _X in test_X:
        test_y.append(check_class(_X))

    initial_classifier.fit(X_initial, y_initial)
    print('Score for initial SVM is: ', initial_classifier.score(test_X, test_y))
    print('Score for final SVM is:', svm_classifier.score(test_X, test_y))

#%%
def fun(point):
    ''' Function to be minimized '''
    global svm_classifier, Gaussian_regressor, C1
    
    # g(x) 
    fun_val = abs((value_prediction_svm(svm_classifier, point))[0])
    
    # U(x) 
    uncertainty = Gaussian_regressor.predict(np.atleast_2d(point), return_std = True)[1][0] 

    # g(x) - C1*U(x)
    return (fun_val - C1 * uncertainty)
#    return fun_val, uncertainty
# %%
'''
Generate X and y data
'''

numpy_randomstate = 3       # set numpy random state if needed
random_state = 15           # random state for lhs sampling
initial_sample_number = 10  # number of initial sampling points


# Check data shape
X = lhs(2, samples= initial_sample_number - 4, random_state=random_state)
X = np.vstack([X, [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])

y = []
for _X in X:
    y.append(check_class(_X))

plt.scatter(X[:,0], X[:,1], c=y, alpha = 0.6)
plt.xlim((-0.1,1.1))
plt.ylim((-0.1,1.1))

#%%
'''
Full loop start if initial data has all classifications (-1 and 1)
'''
# Initial data generation
np.random.seed()
X = lhs(2, samples= initial_sample_number - 4, random_state=random_state)
X = np.vstack([X, [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])

y = []
for _X in X:
    y.append(check_class(_X))

# copy data for plotting
X_initial = X.copy()
y_initial = y.copy()
#%%

# Initial parameter setting
svm_classifier = svm.SVC(kernel='rbf', C = 100000, gamma = 'auto') # initial svm
Gaussian_regressor = GaussianProcessRegressor(normalize_y = True) # initial GPR
initial_svm_classifier = deepcopy(svm_classifier) # copy initial svm for comparison
num_iter = 10       # number of additional sampling
n_optimization = 20 # number of initialization for optimization
bounds = [(0.0, 1.0), (0.0, 1.0)]   # variable bounds
new_points = np.array([]) # initialize new points collection
C1 = 1e1 # weight on the uncertainty

# %%
# Loop start
iter = 0

while iter < num_iter:
    # Fit svm
    svm_classifier.fit(X,y)

    # Calculate g(x) using svm
    continuous_y = svm_classifier.decision_function(X)

    # Train gaussian process regressor using X and continous y
    Gaussian_regressor.fit(X, continuous_y) 

    # if we want to see heatmap of uncertainty, we can activate the next line
#    plot_heatmap_uncertainty(Gaussian_regressor)

    # Optimize |g(x)| - C1 * uncertainty
    opt_x = []
    opt_fun = []
    optimization_sample = lhs(2, 20)
    for i in range(n_optimization):
        np.random.seed()
        opt = minimize(fun, x0 = optimization_sample[i], method = "L-BFGS-B", bounds=bounds)
        opt_x.append(opt.x)
        opt_fun.append(opt.fun)
    new_fun = min(opt_fun)
    new_x = opt_x[np.argmin(opt_fun)]
    
    # Add new_x to the training data
    X = np.vstack([X, new_x])
    if iter == 0 :
        new_points = np.atleast_2d(new_x)
    else:
        new_points = np.vstack([new_points, new_x])
    y.append(check_class(new_x))

    # Print
    print('Iteration {} : Added point x value is {} and function value is {}'.format(iter, new_x, new_fun))
    
    # Add iteration number
    iter += 1

# %%
# Scatter plot generation
plot_scatter_data(svm_classifier, X, y, new_points)

# Plot svm boundary
plot_svm_boundary(svm_classifier, X, y)

# %%
# Testing with random points for score
np.random.seed()
test_svm(1000, svm_classifier, initial_svm_classifier)

###########################################################################
# %%
''' 
Starting Random sampling-based SVM improvement 
'''

X = lhs(2, samples= initial_sample_number + num_iter, random_state=random_state)
y = []
for _X in X:
    y.append(check_class(_X))

X_initial = X.copy()
y_initial = y.copy()

# Initial setting
random_svm_classifier = svm.SVC(kernel='rbf', C = 100000, gamma = 'auto')
random_initial_svm_classifier = deepcopy(random_svm_classifier)

# Train svm with random sampling
func = fun
bounds = [(0.0, 1.0), (0.0, 1.0)]
iter = 0
# new_points = np.array([])
# while iter < num_iter:
#     # Fit svm
#     # get the new x and y
#     np.random.seed()
#     x_sample = np.random.random(2)
#     X = np.vstack([X, x_sample])
#     if iter == 0 :
#         new_points = np.atleast_2d(x_sample)
#     else:
#         new_points = np.vstack([new_points, x_sample])
#     y.append(check_class(x_sample))
#     print('Iteration {} : Added point x value is {}'.format(iter, x_sample))
#     iter += 1

random_svm_classifier.fit(X,y)

# Scatter plot
plot_scatter_data(random_svm_classifier, X, y)

# Boundary plot
plot_svm_boundary(random_svm_classifier, X, y)

# Test
test_svm(1000, random_svm_classifier, random_initial_svm_classifier)

# %%
