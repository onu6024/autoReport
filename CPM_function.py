"""
Created on Wed Feb 15 11:55:30 2023

@author: ninewatt

piecewise regression 코드 레퍼런스

https://zephyrus1111.tistory.com/335

"""

import numpy as np

class StepFunction():
    def __init__(self, lower_knot, upper_knot):
        assert lower_knot < upper_knot
        self.lower_knot = lower_knot
        self.upper_knot = upper_knot
        
    def get_value(self, x):
        lower_knot = self.lower_knot
        upper_knot = self.upper_knot
        lower_step = np.piecewise(x, [x<=lower_knot, x>lower_knot], [1, 0])
        upper_step = np.piecewise(x, [x<=upper_knot, x>upper_knot], [1, 0])
        return int(upper_step - lower_step)
    
class PiecewiseRegression():    
    def __init__(self, knots, order=1):
        assert order<=3 and order>=0
        self.order = order
        self.knots = knots
        self.coef = None
        self.basis = None
#         self.model = None
        self.coef_ = None
        self.intercept_ = None
        self.fitted_values = None
        self.new_X = None
        self.constraint = None
        
    def fit(self, X, y, constraint=False):
        from sklearn.linear_model import LinearRegression
        from scipy.linalg import lapack
        self.constraint = constraint
        self.make_bases()
        
        new_X = self.make_design_matrix(X)
        self.new_X = new_X
        if not constraint:
            model = LinearRegression(fit_intercept=False).fit(new_X, y)
            self.coef_ = model.coef_
            self.fitted_values = model.predict(new_X)
        else:
            A = self.make_constraint_matrix()
            self.coef_ = lapack.dgglse(new_X, A, y, np.zeros(len(self.knots)))[3]
            self.fitted_values = new_X.dot(self.coef_)
        return self
    
    def make_constraint_matrix(self):
        row = []
        knots = self.knots
        order = self.order
        for k in range(len(knots)):
            knot = knots[k]
            temp_rows = np.zeros((order+1)*(len(knots)+1))
            temp_knot_poly = np.array([knot**o for o in range(order+1)])
            temp_rows[(order+1)*k:(order+1)*(k+2)] = np.concatenate([temp_knot_poly, -temp_knot_poly])
            row.append(temp_rows)
 
        return np.row_stack(row)
    
    def make_bases(self):
        import numpy as np
        knots = self.knots     
        basis = []
        num_knots = len(knots)
        for i in range(num_knots+1):
            if i == 0:
                basis.append(StepFunction(-(np.infty), knots[i]))
            elif i == num_knots:
                basis.append(StepFunction(knots[i-1], np.infty))
            else:
                basis.append(StepFunction(knots[i-1], knots[i]))
                
        self.basis = basis
     
     
    def make_design_matrix(self, X):
        num_intervals = len(self.basis)
        order = self.order
        col = []
        for b in self.basis:
            for o in range(order+1):
                col.append(np.array([(x**(o))*b.get_value(x) for x in X.flatten()]))
 
        return np.column_stack(col)
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        order = self.order
        x = x[0]
        temp_array = [b.get_value(x) for b in self.basis]
        temp_idx = temp_array.index(1)
        x_array = np.array([x**(o) for o in range(order+1)])
        target_coef = self.coef_[(order+1)*temp_idx:(order+1)*(temp_idx+1)]
        return x_array.dot(target_coef)
    



















