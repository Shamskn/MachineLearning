from abc import ABCMeta, abstractmethod

class Regression(metaclass=ABCMeta):
    '''
    Abstract class for regression
    '''
    @abstractmethod
    def __init__(self, *args, **kwargs):
        self.fit_intercept = kwargs.pop('fit_intercept')

    @abstractmethod
    def fit(self, X, t):
        pass

    @abstractmethod
    def predict(self, X):
        pass

