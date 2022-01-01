import cupy as np
from utils import sigmoid, relu, sigmoid_backward, relu_backward

np.random.seed(1)

SIGMOID = "sigmoid"
RELU = "relu"


# 2-layer Neural Network
# model's structure: LINEAR -> RELU -> LINEAR -> SIGMOID
# initialize parameters for a 2-layer network
def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    :param n_x: size of input layer
    :param n_h: size of hidden layer
    :param n_y: size of output layer
    :return: a dictionary containing the following:
        W1: weight matrix of shape (n_h,n_x)
        b1: bias vector of shape (n_h,1)
        W2: weight matrix of shape (n_y,n_h)
        b2: bias vector of shape (n_y,1)
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters


# initialize parameters for an L-layer network
def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.1
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# Forward propagation module
# linear forward module
def linear_forward(A_prev, W, b):
    """
    Arguments:
    :param A_prev: activation values from previous layer
    :param W: Weight matrix of current layer
    :param b: bias vector of current layer
    :return:
        Z: linear activation values of current layer
        cache: a dictionary containing current layers' weights, biases and previous layer activation
                The reason for doing so is that these values could be useful for backpropagation step
    """
    Z = np.dot(W, A_prev) + b
    cache = (A_prev, W, b)
    return Z, cache


# linear activation forward, where we compute linear function followed by an activation
def linear_activation_forward(A_prev, W, b, activation):
    A = 0
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == SIGMOID:
        A = sigmoid(Z)
    elif activation == RELU:
        A = relu(Z)
    activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache


# L-layered model which has L-1 RELU activations followed by a SIGMOID activation
def L_model_forward(X, parameters):
    # number of layers in the network
    L = len(parameters) // 2

    caches = []
    A = X
    # L-1 times RELU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], RELU)
        caches.append(cache)

    # last layer is sigmoid
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], SIGMOID)
    caches.append(cache)

    return AL, caches


# compute cost using cross-entropy loss function
def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    cost = np.squeeze(cost)
    return cost


# Backward propagation module
# for a layer L, when we  have dZL, cache; now we have to compute dA_prev,dW,db
def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


# in backward step, now we compute dA_prev, dW, db with the help of linear_backward and cache
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    dZ = 0
    if activation == SIGMOID:
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == RELU:
        dZ = relu_backward(dA, activation_cache)
    return linear_backward(dZ, linear_cache)


# L-model backward propagation
def L_model_backward(AL, Y, caches):
    # initialize empty dictionary for storing gradients
    grads = {}

    # number of layers
    L = len(caches)

    # first calculate the gradient of last/output layer
    grads["dA"+str(L)] = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(grads["dA" + str(L)], caches[L - 1], SIGMOID)

    for l in reversed(range(1, L)):
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward(
            grads["dA" + str(l)], caches[l - 1], RELU)

    # remove dA0 as it's not necessary
    grads.pop("dA0")

    return grads


# update parameters using gradient descent
def update_parameters(parameters, grads, learning_rate):
    # number of layers
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * grads["dW" + str(l)])
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * grads["db" + str(l)])

    return parameters
