import numpy as np
from classification.multiclass import MultipleLogisticRegression as mlr
from classification.logistic_regression import LogisticRegression

t = np.array([0,1,1,2]).reshape(-1,1)
w = np.array([2., 3., 1.]).reshape(-1,1)
X = np.array([[1., 2., 3.],
               [2., 4., 5.],
               [1., 3., 7.],
              [4., 2., 7.]])

x1 = np.random.normal(size=(100, 2))
x1 += np.array([-5, -5])
x2 = np.random.normal(size=(100, 2))
x2 += np.array([5, -5])
x3 = np.random.normal(size=(100, 2))
x3 += np.array([0, 5])
x_train = np.vstack((x1, x2, x3))
np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
from datetime import date, datetime
datetime.now().date()