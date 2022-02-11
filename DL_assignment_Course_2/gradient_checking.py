import cupy as np

from DL_assignment_Course_2.gc_utils import relu, sigmoid, dictionary_to_vector, gradients_to_vector, \
    vector_to_dictionary
from DL_assignment_Course_2.init import load_2D_dataset
from deep_network_step_by_step.deep_model import update_parameters
from init import load_2D_dataset
from matplotlib import pyplot as plt
import math


def forward_propagation_n(X, Y, parameters, compute_cost=True):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.

    Arguments:
    X -- training set for m examples
    Y -- labels for m examples
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)

    Returns:
    cost -- the cost function (logistic cost for one example)
    """

    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    cost = 0.0
    if compute_cost is True:
        logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
        cost = 1. / m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache


def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()

    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.array(np.int64(A2.get() > 0)))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.array(np.int64(A1.get() > 0)))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients


def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation_n

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    x -- input datapoint, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
        # "_" is used because the function you have to outputs two parameters but we only care about the first one
        ### START CODE HERE ### (approx. 3 lines)
        thetaplus = np.copy(parameters_values)  # Step 1
        thetaplus[i][0] += epsilon  # Step 2
        J_plus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaplus, old_parameters=parameters))  # Step 3
        ### END CODE HERE ###

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
        ### START CODE HERE ### (approx. 3 lines)
        thetaminus = np.copy(parameters_values)  # Step 1
        thetaminus[i][0] -= epsilon  # Step 2
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(thetaminus, old_parameters=parameters))  # Step 3
        ### END CODE HERE ###

        # Compute gradapprox[i]
        ### START CODE HERE ### (approx. 1 line)
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2.0 * epsilon)
        ### END CODE HERE ###

    # Compare gradapprox to backward propagation gradients by computing difference.
    ### START CODE HERE ### (approx. 1 line)
    numerator = np.linalg.norm(grad - gradapprox)  # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)  # Step 2'
    difference = numerator / denominator  # Step 3'
    ### END CODE HERE ###

    if difference > 2e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

def model(X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True):
    '''
    Implements a three layer NN: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
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
    layer_dims = [X.shape[0], 5, 3, 1]

    # initialize parameters
    np.random.seed(3)
    for l in range(1, len(layer_dims)):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * math.sqrt(
            2. / layer_dims[l - 1])
        parameters["b" + str(l)] = np.zeros(shape=(layer_dims[l], 1))

    # gradient descent
    for i in range(num_iterations):
        cost, caches = forward_propagation_n(X, Y, parameters)
        if print_cost and i is not 0 and i % 1000 == 0:
            costs.append(np.ndarray.get(cost))
            print("Cost after iteration %i: %f" % (i, cost))
            gradient_check_n(parameters, grads, X, Y, epsilon=1e-4)
        grads = backward_propagation_n(X, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show(block=True)
    return parameters

def predictions(X, Y, parameters):
    _, cache = forward_propagation_n(X, Y, parameters, compute_cost=False)

    # convert probabilities into actual predictions
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    Y_predictions = A3
    Y_predictions[(Y_predictions > 0.5)] = 1
    Y_predictions[(Y_predictions <= 0.5)] = 0

    return Y_predictions

def plot_decision_boundary(modl, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = modl(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    cs = plt.contourf(xx.get(), yy.get(), Z.get(), cmap=plt.cm.Spectral)
    plt.colorbar(cs)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :].get(), X[1, :].get(), c=y.get(), cmap=plt.cm.Spectral)
    plt.show(block=True)


train_X, train_Y, test_X, test_Y = load_2D_dataset()
train_X, train_Y, test_X, test_Y = np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)

params = model(train_X, train_Y, learning_rate=0.1, num_iterations=30000)
# training accuracy
Y_prediction_train = predictions(train_X, train_Y, params)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_Y)) * 100))

# test accuracy
Y_prediction_test = predictions(test_X, test_Y, params)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_Y)) * 100))

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predictions(x.T, train_Y, params), train_X, train_Y)
