# %%
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from pyDOE2 import *

# %%
def regr_fun(x):
    global regr
    return float(regr.predict(np.atleast_2d(x))[0])
# %%

# %%
def fun(x):
    return (np.sin(2*x))**2 

#X = np.atleast_2d(np.random.randn(500)).T
X = np.atleast_2d(4 * lhs(n=1, samples= 100, criterion= 'maximin') - 2.0)

y = np.array(list(map(fun, X))).ravel()
plt.scatter(X[:,0], y)
# %%
#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(hidden_layer_sizes=(100,100,100), random_state=1, max_iter=500).fit(X, y)
regr.score(X,y)

# %%
# %%
x_plot = np.linspace(-2,2,100)
y_plot = []
for _x in x_plot:
    y_plot.append(regr_fun(_x))
plt.plot(x_plot, y_plot)

# %%
x = 4 * (np.random.random(10) - 0.5 )
x_opt = []
y_opt = []

for _x in x:
    opt = minimize(fun= regr_fun, x0 = np.array([_x]), method = 'l-bfgs-b', bounds = [(-2,2)])
    x_opt.append(opt.x)
    y_opt.append(opt.fun)

# %%
opt# %%

# %%
