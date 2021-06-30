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

import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import plotly 
# %%
''' Functions definition '''
def value_prediction_svm(svm_classifier, point):    
    ''' calculate g(x) of point '''
    value_prediction = svm_classifier.decision_function(np.atleast_2d(point))   
    return value_prediction

def check_class(x):
    ''' check classification of data x '''
#    if (x[0] - 0.5)**2 +  (x[1] - 0.5)**2 <= 0.2 **2:
#    if (x[0]-0.5)**2 + (x[1]-0.5)**2 + (x[2]-0.5)**2 <= 0.3 ** 2:
    x_bar = 15 * x - 5
    fun_val = 1/(3.755 * 10**5) *(sum(100 * (x_bar[i] - x_bar[i+1]**2)**2 + (1-x_bar[i])**2 for i in range(2)) - 3.827 * 10 **5)
    if fun_val <= 0.5:
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

def plot_scatter_data(svm_classifier, X,y, new_points):
    ''' scatter plot for data '''
    plt.scatter(X[:initial_sample_number,0], X[:initial_sample_number,1], c=y[:initial_sample_number], s=30, alpha = 0.3)
    #plt.plot(x,y_hyperplane)
    plt.scatter(new_points[:,0], new_points[:,1], s=50, c = 'r', marker = '*')
#    plt.scatter(svm_classifier.support_vectors_[:,0], 
#                svm_classifier.support_vectors_[:,1], 
#                s=15, marker='x')

    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.legend(['Initial points', 'new points', 'support vectors'])
    plt.show()

def test_svm(num_test_points, svm_classifier, initial_classifier):
    ''' test final svm accuracy and compare with initial svm '''
    test_X = np.random.random([num_test_points, 3])
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

# %%
'''
Generate X and y data
'''

numpy_randomstate = 3       # set numpy random state if needed
random_state = 11           # random state for lhs sampling
initial_sample_number = 9  # number of initial sampling points
C1 = 1e0 # weight on the uncertainty

# Check data shape
X = np.array([[0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.5, 0.5, 0.5]])
y = []
for _X in X:
    y.append(check_class(_X))

if 1 in y and -1 in y:
    print('Data contains both classifications. Good to go')    
else: 
    raise ValueError('One classification data is missing. Different random state for lhs is needed.')
#%%
'''
Full loop start if initial data has all classifications (-1 and 1)
'''
# Initial data generation
# np.random.seed()
# X = lhs(3, samples= initial_sample_number, random_state=random_state)
# y = []
# for _X in X:
#     y.append(check_class(_X))

# copy data for plotting
X_initial = X.copy()
y_initial = y.copy()
#%%

# Initial parameter setting
svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42) # initial svm
Gaussian_regressor = GaussianProcessRegressor(normalize_y = True) # initial GPR
initial_svm_classifier = deepcopy(svm_classifier) # copy initial svm for comparison
num_iter = 20       # number of additional sampling
n_optimization = 10 # number of initialization for optimization
bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]   # variable bounds
new_points = np.array([]) # initialize new points collection


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
    # plot_heatmap_uncertainty(Gaussian_regressor)

    # Optimize |g(x)| - C1 * uncertainty
    opt_x = []
    opt_fun = []
    for i in range(n_optimization):
        np.random.seed()
        opt = minimize(fun, x0 = np.random.rand(3), method = "L-BFGS-B", bounds=bounds)
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
# plot_scatter_data(svm_classifier, X, y, new_points)

# Plot svm boundary
# plot_svm_boundary(svm_classifier, X, y)

# %%
# Testing with random points for score
np.random.seed()
test_svm(1000, svm_classifier, initial_svm_classifier)

###########################################################################
# %%
''' 
Starting LHS + random sampling-based SVM 
'''

X_random = lhs(3, samples= initial_sample_number + num_iter, random_state=random_state)
y_random = []
for _X in X_random:
    y_random.append(check_class(_X))

X_random_initial = X_random.copy()
y_random_initial = y_random.copy()

# Initial setting
random_svm_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42)
random_initial_svm_classifier = deepcopy(random_svm_classifier)

# # Train svm with random sampling
# func = fun
# bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
# iter = 0
# new_points = np.array([])
# while iter < num_iter:
#     # Fit svm
#     # get the new x and y
#     np.random.seed()
#     x_sample = np.random.random(3)
#     X = np.vstack([X, x_sample])
#     if iter == 0 :
#         new_points = np.atleast_2d(x_sample)
#     else:
#         new_points = np.vstack([new_points, x_sample])
#     y.append(check_class(x_sample))
#     print('Iteration {} : Added point x value is {}'.format(iter, x_sample))
#     iter += 1

random_svm_classifier.fit(X_random,y_random)

# Scatter plot
#plot_scatter_data(random_svm_classifier, X, y, new_points)

# Boundary plot
#plot_svm_boundary(random_svm_classifier, X, y)

# Test
test_svm(1000, random_svm_classifier, random_initial_svm_classifier)


#%%
df = pd.DataFrame(X[:9])

df['class'] = y[:9]
df.columns = ['x1', 'x2', 'x3','class']
df_new = pd.DataFrame(new_points)
#df_new['class'] = 20 * np.ones(new_points.shape[0])
df_new['class'] = y[9:]

df_new.columns = ['x1', 'x2', 'x3','class']

ppp = pd.DataFrame([np.random.random(3), [0.5, 0.5, 0.5]])
ppp.columns = ['x1','x2','x3']
ppp = ppp.drop([0], axis = 0)

trace1 = go.Scatter3d(
    x=df['x1'],
    y=df['x2'],
    z=df['x3'],
    mode='markers',
    name = 'initial points',
    marker=dict(
        size=6,
        color=df['class'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)


trace2 = go.Scatter3d(
    x=ppp['x1'],
    y=ppp['x2'],
    z=ppp['x3'],
    mode='markers',
    name = 'True boundary',    
    marker=dict(
        size = 90,
        opacity = 0.2,
        color = 0.1
    )
)

trace3 = go.Scatter3d(
    x=df_new['x1'],
    y=df_new['x2'],
    z=df_new['x3'],
    mode='markers',
    name = 'New samples',
    marker=dict(
        size=6,
        color=df_new['class'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1, trace3, trace2]
layout = go.Layout(
    margin = dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data = data, layout=layout)
plotly.offline.iplot(fig, filename='simple-3d-scatter')

# %%
df_random = pd.DataFrame(X_random)
df_random['class'] = y_random
df_random.columns = ['x1', 'x2', 'x3','class']

ppp_random = pd.DataFrame([np.random.random(3), [0.5, 0.5, 0.5]])
ppp_random.columns = ['x1','x2','x3']
ppp_random = ppp_random.drop([0], axis = 0)

trace1_random = go.Scatter3d(
    x=df_random['x1'],
    y=df_random['x2'],
    z=df_random['x3'],
    mode='markers',
    name = 'Samples',
    marker=dict(
        size=6,
        color=df_random['class'],                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

trace2_random = go.Scatter3d(
    x=ppp_random['x1'],
    y=ppp_random['x2'],
    z=ppp_random['x3'],
    mode='markers',
    name='True boundary',
    marker=dict(
        sizemode = 'diameter',
        size = 90,
        sizeref = 90,
        opacity = 0.15
    )
)

data_random = [trace1_random, trace2_random]

layout_random = go.Layout(
    margin = dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig_random = go.Figure(data = data_random, layout=layout_random)
plotly.offline.iplot(fig_random, filename='simple-3d-scatter')

# %%