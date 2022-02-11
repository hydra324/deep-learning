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


def relu(x):
    """
    Compute the relu of x
    Arguments:
    x -- A scalar or numpy array of any size.
    Return:
    s -- relu(x)
    """
    s = np.maximum(0, x)

    return s


def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dictionary(theta, old_parameters):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    x, y = old_parameters["W1"].shape
    partition_to = x*y
    parameters["W1"] = theta[:partition_to].reshape((x,y))
    x, y = old_parameters["b1"].shape
    partition_from = partition_to
    partition_to += x*y
    parameters["b1"] = theta[partition_from:partition_to].reshape((x,y))
    x, y = old_parameters["W2"].shape
    partition_from = partition_to
    partition_to += x*y
    parameters["W2"] = theta[partition_from:partition_to].reshape((x,y))
    x, y = old_parameters["b2"].shape
    partition_from = partition_to
    partition_to += x * y
    parameters["b2"] = theta[partition_from:partition_to].reshape((x,y))
    x, y = old_parameters["W3"].shape
    partition_from = partition_to
    partition_to += x * y
    parameters["W3"] = theta[partition_from:partition_to].reshape((x,y))
    x, y = old_parameters["b3"].shape
    partition_from = partition_to
    partition_to += x * y
    parameters["b3"] = theta[partition_from:partition_to].reshape((x,y))

    return parameters


def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta