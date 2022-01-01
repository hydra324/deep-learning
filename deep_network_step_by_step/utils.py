import cupy as np
import h5py
from PIL import Image
from random import shuffle, choice
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.interactive(True)

# IMAGE_SIZE = 256
# CHANNELS = 1

IMAGE_SIZE = 64
CHANNELS = 1

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


def sigmoid_backward(dA, activation_cache):
    """
    Computes the derivative of a sigmoid function at x
    :param dA: dAL
    :param activation_cache: contains Z
    :return: derivative of sigmoid activation
    """
    Z = activation_cache
    A = sigmoid(Z)
    dZ = dA * A * (1 - A)
    return dZ


def relu(x):
    """
    computer the relu activation of x
    :param x: A scalar or a numpy array
    :return: relu activation
    """
    return np.maximum(x, 0)


def relu_backward(dA, activation_cache):
    Z = activation_cache
    # g_prime = Z
    # g_prime[g_prime <= 0] = 0
    # g_prime[g_prime > 0] = 1
    # dZ = dA * g_prime
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


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
    images = np.array([i[0] for i in data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    images = images.reshape(images.shape[0], -1)
    images = images.T
    images = images / 255
    labels = np.array([i[1] for i in data])
    labels = labels.reshape(1, labels.shape[0])
    return images, labels


### Below helper functions for loading data are picked from deeplearning.ai's utils###
def load_h5_data(TRAIN_DATA_PATH, TEST_DATA_PATH):
    train_dataset = h5py.File(TRAIN_DATA_PATH, "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(TEST_DATA_PATH, "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    # classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def load_data_flattened(TRAIN_DATA_PATH, TEST_DATA_PATH):
    train_x_orig, train_y, test_x_orig, test_y = load_h5_data(TRAIN_DATA_PATH,TEST_DATA_PATH)
    # Reshape the training and test examples
    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten / 255.
    test_x = test_x_flatten / 255.
    print("train_x's shape: " + str(train_x.shape))
    print("test_x's shape: " + str(test_x.shape))
    return train_x, train_y, test_x, test_y
