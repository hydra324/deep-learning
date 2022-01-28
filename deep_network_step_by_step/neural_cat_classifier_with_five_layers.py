import matplotlib.pyplot as plt

from utils import *
from deep_model import *

# Now load dataset
# train data
# IMAGE_DIRECTORY = '../Cat Classifier/data/training_set'
# training_images, training_labels = load_and_flatten_data_set(IMAGE_DIRECTORY)
# #
# # # test data
# IMAGE_DIRECTORY = '../Cat Classifier/data/test_set'
# test_images, test_labels = load_and_flatten_data_set(IMAGE_DIRECTORY)

TRAIN_H5_DATA_PATH = '../Cat Classifier/data/train_catvnoncat.h5'
TEST_H5_DATA_PATH = '../Cat Classifier/data/test_catvnoncat.h5'
training_images, training_labels, test_images, test_labels, mean, var = load_data_flattened(TRAIN_H5_DATA_PATH, TEST_H5_DATA_PATH)


# Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
def L_layer_model(X, Y, layer_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, lamda=0.0):
    np.random.seed(1)
    parameters = initialize_parameters_deep(layer_dims)

    costs = []
    m = X.shape[1]
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        cost += compute_regularized_cost(parameters, m, lamda)
        if print_cost and i % 100 == 0:
            costs.append(np.ndarray.get(cost))
            print("Cost after iteration %i: %f" % (i, cost))
        grads = L_model_backward(AL, Y, caches, parameters, lamda)
        parameters = update_parameters(parameters, grads, learning_rate)

    plt.plot(costs)
    plt.show(block=True)
    return parameters


# prediction network that runs a forward pass and outputs predictions
def L_layer_model_predict(X, parameters):
    # run forward pass
    AL, _ = L_model_forward(X, parameters)

    # convert probabilities into actual predictions
    Y_predictions = AL
    Y_predictions[(Y_predictions > 0.5)] = 1
    Y_predictions[(Y_predictions <= 0.5)] = 0

    return Y_predictions


# layer dimensions for a 5-layer model
IMAGE_SIZE = 64
CHANNELS = 3
layers_dims = [IMAGE_SIZE * IMAGE_SIZE * CHANNELS, 20, 7, 5, 1]

params = L_layer_model(X=training_images, Y=training_labels, layer_dims=layers_dims, learning_rate=0.07,
                       print_cost=True, num_iterations=3000, lamda=0.9)

# training accuracy
Y_prediction_train = L_layer_model_predict(training_images, params)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - training_labels)) * 100))

# test accuracy
Y_prediction_test = L_layer_model_predict(test_images, params)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_labels)) * 100))

np.random.seed(None)
viz_images = test_images.T.reshape(test_images.shape[1], IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
visualize_img(viz_images, test_labels.T, Y_prediction_test.T, np.random.random_integers(0, len(viz_images)))
