from regression.regression import Regression
import numpy as np
from utils.prepare_inputs import check_X_y, convert_array


class BayesianRegression(Regression):
    '''
    Bayesian Linear Regression lets us introduce prior information in a more principled manner and include in the
    model how certain we are that our solution is correct.

            p(w|t,X) = N(w|m_N, S_N)
    where,      m_N = beta * S_N • X.T •t
        and     S_N ^(-1) = alpha*Ident + beta * X.T • X
    '''
    def __init__(self, fit_intercept=False,alpha=0.1, beta=9.0):
        super(BayesianRegression, self).__init__(fit_intercept=fit_intercept)
        self.alpha = alpha
        self.beta = beta
        self.mu = None      #mean
        self.sigma2 = None      #variance

    def fit(self, X, t):
        '''
        :param X: array of (n_samples, n_features)
                  independent variables
        :param t: array of (n_sample, n_targets)
                  target variables
        :return: self
        '''
        X,t = check_X_y(X, t)

        S_N = np.linalg.inv(self.alpha * np.eye(np.size(X, 1)).T + self.beta * X.T @ X)
        m_N = self.beta * S_N @ X.T @ t

        self.covariance = S_N
        self.w = m_N

        self._intercept = self.w[0]

        return self

    def predict(self, X):
        X = convert_array(X)

        y = X @ self.w
        self.y_std = 1/self.beta + np.sum(X @ self.covariance @ X.T, axis=1)
        return y
