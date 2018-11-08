import numpy as np
from utils.prepare_inputs import check_X_y, convert_array
from regression.regression import Regression
from utils.preprocessing import get_intercept

class LinearRegression(Regression):
    '''
    y(X, w) = X @ w

    [y_1]   [1  x_11 .. x_1n][w_0]
    [y_2]   [1  x_21 .. x_2n][w_1]
    [ : ] = [:   :  ..   :  ][ : ]
    [ : ]   [:   :  ..   :  ][ : ]
    [y_n]   [1  x_k1 .. x_kn][w_n]

    '''
    def __init__(self, fit_intercept=True):
        super(LinearRegression, self).__init__(fit_intercept=fit_intercept)

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

        self.w = np.linalg.pinv(X) @ t

        self.variance = np.mean(np.square(X @ self.w - t))
        self._intercept = self.w[0] if self.fit_intercept else 0

        return self

    def predict(self, X):

        X = convert_array(X)

        y = self.w[0] + X @ self.w[1:] if self.fit_intercept else X @ self.w
        self.y_std = np.sqrt(self.variance) + np.zeros_like(y)
        return y
