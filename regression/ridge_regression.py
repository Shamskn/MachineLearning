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

        :param X: array of (n_samples, n_features)
                  independent variables
        :param t: array of (n_sample, n_targets)
                  target variables
        :return: self
        '''
        X, t = check_X_y(X, t)
        if self.fit_intercept:
            X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))

        id_mtrx = np.eye(np.size(X, 1))
        self.w = np.linalg.solve(self.alpha * id_mtrx + X.T @ X, X.T @ t)
        self._intercept = self.w[0] if self.fit_intercept else 0

        return self


    def predict(self, X):

        X = convert_array(X)

        y = self.w[0] + X @ self.w[1:] if self.fit_intercept else X @ self.w
        return y
