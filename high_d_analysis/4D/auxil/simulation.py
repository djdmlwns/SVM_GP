# %%
import numpy as np

class Simulation():
    '''
    This needs to be modified 
    '''
    def __init__(self, x, **kwargs):
        self.result = None
        self.x = x
        with open('./data/input.txt', 'w') as f:
            f.write(str(x))


    def run(self):
        # use data for simulation
#        data = self.x 
        self.send_data()
        
        result_text = self.retrieve_result()
        result = self.postprocessing_result(result_text)        
        self.result = result


    def send_data(self):
        # send text file to server for simulation
        # adequate formating should be known in the future
        print('Data is sent')
        pass


    def retrieve_result(self):
        # read result text file from server
        with open('./data/output.txt', 'r') as f:
            lines = f.read()
        return lines


    def postprocessing_result(self, result_text):
        # postprocess text to be used in Python
        processed_result = result_text
        return processed_result 