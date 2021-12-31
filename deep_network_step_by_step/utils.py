import cupy as np
from PIL import Image
from random import shuffle, choice
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.interactive(True)

IMAGE_SIZE = 256


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

