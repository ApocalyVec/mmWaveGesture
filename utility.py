import numpy as np
import math


# FROM https://www.geeksforgeeks.org/linear-regression-python-implementation/
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def get_outliers(array, m=2.):
    """
    :param array: list of values
    :return the list of outliers

    """
    data = np.asarray(array)

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m].tolist()


def distance_to_origin(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    # rtn = 0.0
    # for value in point:
    #     rtn = rtn + math.pow(value, 2)
    # return math.sqrt(rtn)
