import numpy as np

class Simulation():
    def __init__(self, **kwargs):
        self.result = None
        self.run()

    def run(self, x):
        result = np.random.randint(-1,2)
        if result == 0 :
            result = -1
        self.result = result
