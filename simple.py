# %%
import numpy as np

class simpleclass():
    def __init__(self, X, y):
        self.X = X # X is a numpy array
        self.y = y # y is a list

    def train(self):
        new_X = np.array([1,1])
        self.X = np.vstack([self.X, new_X])

sc = simpleclass(np.random.random([2,2]), [1,1])

sc.train()
print(sc.X)
