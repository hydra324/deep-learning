from deep_network_step_by_step.deep_model import L_model_forward, compute_cost, L_model_backward, update_parameters, \
    linear_activation_forward, SIGMOID, RELU
from init import load_2D_dataset, plot_decision_boundary
from matplotlib import pyplot as plt
import cupy as np
import math

from utils import sigmoid, relu, sigmoid_backward, relu_backward

train_X, train_Y, test_X, test_Y = load_2D_dataset()
train_X, train_Y, test_X, test_Y = np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        AL, caches = L_model_forward_with_dropout(X, vector_to_dictionary(theta=theta_plus, old_parameters=parameters),
                                                  keep_prob=1)
        J_plus[i] = compute_cost(AL, Y)

        theta_minus = np.copy(parameters_values)
        theta_plus[i][0] -= epsilon
        AL, caches = L_model_forward_with_dropout(X, vector_to_dictionary(theta=theta_minus, old_parameters=parameters),
                                                  keep_prob=1)
        J_minus[i] = compute_cost(AL, Y)

        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2.0 * epsilon)

    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

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

def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True, lambd=0.0, keep_prob=1):
    '''
    Implements a three layer NN: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    :param keep_prob: the probability of keeping neurons in a drop out regularization scheme.
    :param lambd: the hyperparameter for L2 regularization
    :param X: input data
    :param Y: ground truth label
    :param learning_rate: learning rate for gradient descent
    :param num_iterations: number of iterations to run gradient descent
    :param print_cost: True/False
    :return: parameters learnt my the model
    '''

    grads = {}
    costs = []
    # number of examples
    m = X.shape[1]
    parameters = {}
    layer_dims = [X.shape[0], 20, 3, 1]

    # initialize parameters
    np.random.seed(3)
    for l in range(1, len(layer_dims)):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * math.sqrt(
            2. / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros(shape=(layer_dims[l], 1))

    # gradient descent
    for i in range(num_iterations):
        AL, caches = L_model_forward_with_dropout(X, parameters, keep_prob=keep_prob)
        cost = compute_cost(AL, Y)
        cost += compute_L2_regularized_cost(parameters, m, lambd)
        if print_cost and i is not 0 and i % 1000 == 0:
            costs.append(np.ndarray.get(cost))
            print("Cost after iteration %i: %f" % (i, cost))
            gradient_check_n(parameters, grads, X, Y)
        grads = L_model_backward_with_dropout(AL, Y, caches, parameters, lamda=lambd, keep_prob=keep_prob)
        parameters = update_parameters(parameters, grads, learning_rate)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show(block=True)
    return parameters

def predictions(X, parameters):
    AL, _ = L_model_forward(X, parameters)

    # convert probabilities into actual predictions
    Y_predictions = AL
    Y_predictions[(Y_predictions > 0.5)] = 1
    Y_predictions[(Y_predictions <= 0.5)] = 0

    return Y_predictions

def compute_L2_regularized_cost(parameters, m, lamda):
    '''
    computes L2 regularization cost or the Frobenius norm from the weight parameters
    :param lamda: the hyperparameter
    :param parameters: weights of the network
    :param m: number of training examples
    :return:
    '''
    if lamda == 0:
        return 0
    cost = 0.0
    L = len(parameters) // 2
    for l in range(1, L+1):
        cost += np.sum(np.square(parameters["W"+str(l)]))
    cost *= (lamda/(2*m))
    return cost

def L_model_forward_with_dropout(X, parameters, keep_prob=1):
    # number of layers in the network
    L = len(parameters) // 2

    caches = []
    A = X
    # L-1 times RELU
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward_with_dropout(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], RELU, keep_prob=keep_prob)
        caches.append(cache)

    # last layer is sigmoid
    AL, cache = linear_activation_forward_with_dropout(A, parameters["W" + str(L)], parameters["b" + str(L)], SIGMOID, keep_prob=1)
    caches.append(cache)

    return AL, caches

# linear activation forward, where we compute linear function followed by an activation
def linear_activation_forward_with_dropout(A_prev, W, b, activation, keep_prob=1):
    np.random.seed(1)
    A = 0
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == SIGMOID:
        A = sigmoid(Z)
    elif activation == RELU:
        A = relu(Z)
    D = np.random.rand(A.shape[0], A.shape[1])
    D = D < keep_prob
    A = np.multiply(A, D)
    A = A / keep_prob
    activation_cache = Z
    cache = (linear_cache, activation_cache, D)
    return A, cache

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

# L-model backward propagation
def L_model_backward_with_dropout(AL, Y, caches, parameters, lamda=0.0, keep_prob=1):
    # initialize empty dictionary for storing gradients
    grads = {}

    # number of layers
    L = len(caches)
    m = Y.shape[1]

    # first calculate the gradient of last/output layer
    grads["dA"+str(L)] = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(grads["dA" + str(L)], caches[L - 1], SIGMOID)
    grads["dW" + str(L)] = grads["dW" + str(L)] + (lamda / m) * parameters["W"+str(L)]

    for l in reversed(range(1, L)):
        grads["dA" + str(l)] = np.multiply(grads["dA" + str(l)], caches[l-1][2])
        grads["dA" + str(l)] /= keep_prob
        grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = linear_activation_backward(
            grads["dA" + str(l)], caches[l - 1], RELU)
        grads["dW" + str(l)] = grads["dW" + str(l)] + (lamda / m) * parameters["W" + str(l)]

    # remove dA0 as it's not necessary
    grads.pop("dA0")

    return grads

# in backward step, now we compute dA_prev, dW, db with the help of linear_backward and cache
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache, D = cache
    dZ = 0
    if activation == SIGMOID:
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == RELU:
        dZ = relu_backward(dA, activation_cache)
    return linear_backward(dZ, linear_cache)

# Backward propagation module
# for a layer L, when we  have dZL, cache; now we have to compute dA_prev,dW,db
def linear_backward(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db


params = model(train_X, train_Y, learning_rate=0.1, lambd=0.0, keep_prob=1, num_iterations=30000)
# training accuracy
Y_prediction_train = predictions(train_X, params)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_Y)) * 100))

# test accuracy
Y_prediction_test = predictions(test_X, params)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_Y)) * 100))

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predictions(x.T, params), train_X, train_Y)
