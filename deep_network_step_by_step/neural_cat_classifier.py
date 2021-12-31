import cupy as np
from PIL import Image
from random import shuffle, choice
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from deep_model import *
plt.interactive(True)

IMAGE_SIZE = 256


def visualize_img(images, labels, predictions, index):
    plt.imshow(np.ndarray.get(images[index]), cmap=cm.Greys_r)
    title = "label: " + ("cat" if labels[index] == 1 else "not cat")
    title += " your prediction: " + ("cat" if predictions[index] == 1 else "not cat")
    plt.title(title)
    plt.show(block=True)


def label_img(name):
    if name == 'cats':
        return 1
    elif name == 'notcats':
        return 0


def load_data(IMAGE_DIRECTORY):
    print("Loading images...")
    train_data = []
    directories = next(os.walk(IMAGE_DIRECTORY))[1]

    for dirname in directories:
        print("Loading {0}".format(dirname))
        file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, dirname)))[2]

        for i in range(200):
            image_name = choice(file_names)
            image_path = os.path.join(IMAGE_DIRECTORY, dirname, image_name)
            label = label_img(dirname)
            if "DS_Store" not in image_path:
                img = Image.open(image_path)
                img = img.convert('L')
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
                train_data.append([np.array(img), label])

    return train_data


def load_and_flatten_data_set(IMAGE_DIRECTORY):
    data = load_data(IMAGE_DIRECTORY)
    images = np.array([i[0] for i in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    images = images.reshape(images.shape[0], -1)
    images = images.T
    images = images / 255
    labels = np.array([i[1] for i in data])
    labels = labels.reshape(1, labels.shape[0])
    return images, labels


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

### CONSTANTS DEFINING THE MODEL ####
layers_dims = (IMAGE_SIZE * IMAGE_SIZE, 7, 1)
# train two layer neural net model
params = two_layer_model(X=training_images, Y=training_labels, layer_dims=layers_dims, learning_rate=0.005, print_cost=True)

# training accuracy
Y_prediction_train = two_layer_model_predict(training_images, params)
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - training_labels)) * 100))

# test accuracy
Y_prediction_test = two_layer_model_predict(test_images, params)
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_labels)) * 100))

viz_images = test_images.T.reshape(test_images.shape[1], 256, 256)
visualize_img(viz_images, test_labels.T, Y_prediction_test.T, 0)
