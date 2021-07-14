    def acquisition_function(self, x):
        ''' Objective function to be minimized '''            
        # g(x) 
        fun_val = abs((self.value_prediction_svm(x))[0]) 
        
        # U(x) 
        uncertainty = self.GP_regressor.predict(np.atleast_2d(x), return_std = True)[1][0] 

        return fun_val - self.C1 * uncertainty
