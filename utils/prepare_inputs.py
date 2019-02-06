import numpy as np


def convert_array(array, dtype=np.float64, ensure_2d=True):
    """
    The input array is converted to numpy floats by default

    :param array:
    :param dtype:
    :param ensure_2d:
    :return:
    """
    arr_dtype = getattr(array, "dtype", None)

    if arr_dtype != 'float64':
        array = np.array(array, dtype=dtype)

    if ensure_2d:
        # If input is 1D raise error
        if array.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(array))
    return array


def equal_lengths(*arrays):
    """
    Checks whether all objects in arrays have the same shape or length
    """
    lengths = [len(X) for X in arrays if X is not None]
    uniq = set(lengths)
    if len(uniq) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def check_X_y(X, y, dtype=np.float64):
    X = convert_array(X, dtype=dtype, ensure_2d=True)

    shape = np.shape(y)
    if (len(shape) == 1) or (len(shape) == 2 and shape[1] == 1):
        y = np.array(y, dtype=dtype).reshape(-1, 1)
    else:
        raise ValueError("bad input shape {0}".format(shape))

    equal_lengths(X, y)

    return X, y


def target_is_binary(y):
    return np.array_equal(y, y.astype(bool))
