from deep_network_step_by_step.deep_model import L_model_forward, compute_cost, L_model_backward, update_parameters
from init import load_dataset, plot_decision_boundary
from matplotlib import pyplot as plt
import cupy as np
import math

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()
train_X, train_Y, test_X, test_Y = np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

'''
You will use a 3-layer neural network (already implemented for you). Here are the initialization methods you will experiment with:

Zeros initialization -- setting initialization = "zeros" in the input argument.
Random initialization -- setting initialization = "random" in the input argument. This initializes the weights to large random values.
He initialization -- setting initialization = "he" in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015.
In the next part you will implement the three initialization methods that this model() calls.
'''


def model(X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    '''
    Implements a three layer NN: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    :param X: input data
    :param Y: ground truth label
    :param learning_rate: learning rate for gradient descent
    :param num_iterations: number of iterations to run gradient descent
    :param print_cost: True/False
    :param initialization: flag to choose which initialization ("zeros","random","he")
    :return: parameters learnt my the model
    '''

    grads = {}
    costs = []
    # number of examples
    m = X.shape[1]
    parameters = {}
    layer_dims = [X.shape[0], 15, 10, 5, 1]

    # initialize parameters
    for l in range(1, len(layer_dims)):
        if initialization == "zeros":
            parameters["W" + str(l)] = np.zeros(shape=(layer_dims[l], layer_dims[l - 1]))
            parameters["b" + str(l)] = np.zeros(shape=(layer_dims[l], 1))
        if initialization == "random":
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 2
            parameters["b" + str(l)] = np.zeros(shape=(layer_dims[l], 1))
        if initialization == "he":
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * math.sqrt(
                2. / layer_dims[l - 1])
            parameters["b" + str(l)] = np.zeros(shape=(layer_dims[l], 1))

    # gradient descent
    for i in range(num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        if print_cost and i % 100 == 0:
            costs.append(np.ndarray.get(cost))
            print("Cost after iteration %i: %f" % (i, cost))
        grads = L_model_backward(AL, Y, caches, parameters, 0.0)
        parameters = update_parameters(parameters, grads, learning_rate)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
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


initialization = "he"
# train
params = model(train_X, train_Y, initialization=initialization, num_iterations=25000)

# training accuracy
Y_prediction_train = predictions(train_X, params)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_Y)) * 100))

# test accuracy
Y_prediction_test = predictions(test_X, params)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_Y)) * 100))

plt.title("Model with" + initialization + "initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predictions(x.T, params), train_X, train_Y)
