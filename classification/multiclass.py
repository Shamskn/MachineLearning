import numpy as np
from classification.classification import Classification
from utils.prepare_inputs import convert_array


class MultipleLogisticRegression(Classification):
    """
    A generative model for multiclass classification. The probability is given by the softmax transformation

            p(C_k|X) = y_k(X) = exp(a_k) / sum(exp(a_j)) ∀ j

    where the 'a_k' the activation function is given by:
            a_k = X • w_k

            p(T|w1, . . . ,wK) = prod( prod( y_nk^(t_nk) ), k=1 to K) n=1 to N)

            t_nk = (0, 1, 0 ,...0) where all elements are zero except for element k
    and T is an NxK matrix of target variables with elements t_nk
    """

    def __init__(self, fit_intercept=True, max_iter=100):
        super(MultipleLogisticRegression, self).__init__(fit_intercept=fit_intercept)
        self.max_iter = max_iter

    def fit(self, X, t, learning_rate=0.1):
        """
        Iterative reweighted least squares based on the Newton-Raphson optimisation technique.

                w_new = w_old - H^(-1)•∇E(w)

        where       H = ∇∇E(w) = X.T•R•X
        and         ∇E(w) = X.T•(y - t)

        :param learning_rate:
        :param X: array of (n_samples, n_features)
                  independent variables
        :param t: array of (n_sample, n_targets)
                  target variables
        :return: self
        """
        X = convert_array(X)
        self.n_classes = np.max(t) + 1
        T = np.eye(self.n_classes)[t.flatten()]
        W = np.zeros((np.size(X, 1), self.n_classes)).T
        for _ in range(self.max_iter):
            W_prev = np.copy(W)
            y = self._softmax(X @ W)
            grad = X.T @ (y - T)
            W = W - (learning_rate * grad)
            if np.allclose(W, W_prev):
                break
        self.W = W
        return self

    def _softmax(self, X):
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

    def pred_prob(self, X):
        return self._softmax(X @ self.W)

    def classify(self, X):
        X = convert_array(X)
        proba = self.pred_prob(X)
        y_pred = np.argmax(proba, axis=-1)
        return y_pred
