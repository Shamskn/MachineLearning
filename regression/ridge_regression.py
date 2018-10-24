import numpy as np
from utils.prepare_inputs import check_X_y, convert_array
from regression.regression import Regression


class RidgeRegression(Regression):
    '''
    Regularisation technique is used to control the over-fitting. By adding a penalty term to the error function
    discourage the coefficients from reaching large values.

    This kinda shrinkage method with quadratic reguliser is called Ridge Regression.
    '''
    def __init__(self, fit_intercept=True, alpha = 1.0):
        super(RidgeRegression, self).__init__(fit_intercept=fit_intercept)
        self.alpha = alpha

    def fit(self, X, t):
        '''
        Compute the (Moore-Penrose) pseudo-inverse of a matrix.

        Calculate the generalized inverse of a matrix using its
        singular-value decomposition (SVD)

        :param X: independent variables
        :param t: target variables
        :return: None
        '''
        X, t = check_X_y(X, t)
        self.w = np.linalg.pinv(X) @ t
        self.variance = np.mean(np.square(X @ self.w - t))
        return self


    def predict(self, X):

        X = convert_array(X)

        y = X @ self.w
        self.y_std = np.sqrt(self.variance) + np.zeros_like(y)
        return y