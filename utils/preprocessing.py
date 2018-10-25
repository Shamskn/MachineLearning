import numpy as np

def get_intercept(X, y):
    '''
    Calculating the function intercept
    :param X:
    :param y:
    :return:
    '''
    X_offset = np.average(X, axis=0)
    _X = X - X_offset
    y_offset = np.average(y, axis=0)
    _y = y - y_offset

    # X, y, X_offset, y_offset, X_scale ##
    coef_, _residues, rank_, singular_ = np.linalg.lstsq(_X, _y)

    #Calculating the intercept:
        # b = mean(y) - slope * mean(X)
    _intercept = y_offset - np.dot(X_offset, coef_.T)

    return coef_, _residues, rank_, singular_, _intercept


