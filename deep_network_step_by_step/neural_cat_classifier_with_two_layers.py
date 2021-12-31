from deep_model import *
from utils import *

plt.interactive(True)


# two layer model
# LINEAR -> RELU -> LINEAR -> SIGMOID
def two_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    n_x, n_h, n_y = layer_dims
    parameters = initialize_parameters(n_x, n_h, n_y)

    costs = []
    for i in range(num_iterations):
        # layer 1 is LINEAR->RELU
        A1, cache1 = linear_activation_forward(X, parameters["W1"], parameters["b1"], RELU)
        # layer 2 is LINEAR->SIGMOID
        A2, cache2 = linear_activation_forward(A1, parameters["W2"], parameters["b2"], SIGMOID)
        cost = compute_cost(A2, Y)
        if print_cost and i % 100 == 0:
            costs.append(np.ndarray.get(cost))
            print("Cost after iteration %i: %f" % (i, cost))
        dA2 = -(np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, SIGMOID)
        _, dW1, db1 = linear_activation_backward(dA1, cache1, RELU)
        grads = {"dW2": dW2, "db2": db2, "dW1": dW1, "db1": db1}
        parameters = update_parameters(parameters, grads, learning_rate)
    plt.plot(costs)
    plt.show(block=True)
    return parameters


def two_layer_model_predict(X, parameters):
    # run forward pass
    A1, _ = linear_activation_forward(X, parameters["W1"], parameters["b1"], RELU)
    A2, _ = linear_activation_forward(A1, parameters["W2"], parameters["b2"], SIGMOID)

    # convert probabilities into actual predictions
    Y_predictions = A2
    Y_predictions[(Y_predictions > 0.5)] = 1
    Y_predictions[(Y_predictions <= 0.5)] = 0

    return Y_predictions


# Now load dataset
# train data
IMAGE_DIRECTORY = '../Cat Classifier/data/training_set'
training_images, training_labels = load_and_flatten_data_set(IMAGE_DIRECTORY)

# test data
IMAGE_DIRECTORY = '../Cat Classifier/data/test_set'
test_images, test_labels = load_and_flatten_data_set(IMAGE_DIRECTORY)

# d = model(training_images, training_labels, test_images, test_labels, num_iterations=2000, learning_rate=0.009,
#           print_cost=True)

# layer dimensions defining the model
layers_dims = (IMAGE_SIZE * IMAGE_SIZE, 7, 1)
# train two layer neural net model
params = two_layer_model(X=training_images, Y=training_labels, layer_dims=layers_dims, learning_rate=0.005,
                         print_cost=True)

# training accuracy
Y_prediction_train = two_layer_model_predict(training_images, params)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - training_labels)) * 100))

# test accuracy
Y_prediction_test = two_layer_model_predict(test_images, params)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_labels)) * 100))

viz_images = test_images.T.reshape(test_images.shape[1], 256, 256)
visualize_img(viz_images, test_labels.T, Y_prediction_test.T, 0)
