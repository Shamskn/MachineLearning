import numpy as np
from classification.classification import Classification
from utils.prepare_inputs import target_is_binary, convert_array


class LogisticRegression(Classification):
    """
    A logistic regression model is used to solve to two-class classification problems.
    The posterior probability is the logistic sigmoid acting on the linear function of feature
    vectors.
        p(C_1|X) = y(X) = σ(X•w) = 1 - p(C_2|X)

        where σ(a) =1/1 + exp(−a)

    """

    def __init__(self, fit_intercept=True, max_iter=100):
        super(LogisticRegression, self).__init__(fit_intercept=fit_intercept)
        self.max_iter = max_iter

    def fit(self, X, t):
        """
        Iterative reweighted least squares based on the Newton-Raphson optimisation technique.

                w_new = w_old - H^(-1)•∇E(w)

        where       H = ∇∇E(w) = X.T•R•X
        and         ∇E(w) = X.T•(y - t)

        :param X: array of (n_samples, n_features)
                  independent variables
        :param t: array of (n_sample, n_targets)
                  target variables
        :return: self
        """
        X = convert_array(X, t)

        if not target_is_binary(t):
            raise ValueError("Target variable must be binary classification")

        w = np.zeros(np.size(X, 1)).reshape(-1, 1)

        for i in range(self.max_iter):
            w_i = np.copy(w)
            y = self._sigmoid(X @ w)

            # gradient of the cross entropy error function with respect to 'w'
            grad_err = X.T @ (y - t)

            # Second derivative of the gradient gives us the Hessian;
            # Where R is the NxN diagonal matric with R_nn = y_n(1 - y_n)
            R = np.diag((y * (1 - y)).ravel())
            H = X.T @ R @ X

            try:
                Hinv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                break
            # Newton-Raphson update for the vector 'w'
            w = w - Hinv.dot(grad_err)
            if np.allclose(w, w_i):
                break
        self.w = w
        return self

    def _sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def pred_prob(self, X):
        X = convert_array(X)
        y = self._sigmoid(X @ self.w)
        return y

    def classify(self, X, threshhold=0.5):
        X = convert_array(X)
        proba = self.pred_prob(X)
        y_pred = (proba > threshhold).astype(np.int).flatten()
        return y_pred
