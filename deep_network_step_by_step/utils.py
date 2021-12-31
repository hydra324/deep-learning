import cupy as np


def sigmoid(x):
    """
    Compute the sigmoid of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- sigmoid(x)
    """
    s = 1 / (1 + np.exp(-x))
    return s


def sigmoid_prime(activation_cache):
    """
    Computes the derivative of a sigmoid function at x
    :param activation_cache: contains A and Z
    :return: derivative of sigmoid activation
    """
    A, Z = activation_cache
    g_prime = np.multiply(A, 1 - A)
    return g_prime


def relu(x):
    """
    computer the relu activation of x
    :param x: A scalar or a numpy array
    :return: relu activation
    """
    return np.maximum(x, 0)


def relu_prime(activation_cache):
    A, Z = activation_cache
    g_prime = Z
    g_prime[(g_prime <= 0)] = 0
    g_prime[(g_prime > 0)] = 1
    return g_prime
