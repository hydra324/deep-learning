#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 12:20:44 2021

@author: akhil
"""

from PIL import Image
from random import shuffle, choice
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMAGE_SIZE = 256


def visualize_img(images, labels, predictions, index):
    plt.imshow(images[index], cmap=cm.Greys_r)
    title = "label: " + ("cat" if labels[index] == 1 else "not cat")
    title += " your prediction: " + ("cat" if predictions[index] == 1 else "not cat")
    plt.title(title)


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


def sigmoid(z):
    # define sigmoid activation function
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    # initialize weights and bias to be zero
    w = np.zeros(shape=(dim, 1))
    b = 0.0
    return w, b


# Now we do forward and backward propagation
def propagate(w, b, X, Y):
    m = X.shape[1]

    # forward propagate from X to cost
    epsilon = 1e-5
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum((Y * np.log(A + epsilon)) + ((1 - Y) * (np.log(1 - A + epsilon))))

    # backward propagation to find gradients
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost


# Now we run a gradient descent algorithm to find the
# optimum weights and bias for a minimum cost value
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # record cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # compute logistic regression probabilities
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # convert probabilities to actual predictions
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    assert (Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# Now load dataset
# train data
IMAGE_DIRECTORY = './Cat Classifier/data/training_set'
training_images, training_labels = load_and_flatten_data_set(IMAGE_DIRECTORY)

# test data
IMAGE_DIRECTORY = './Cat Classifier/data/test_set'
test_images, test_labels = load_and_flatten_data_set(IMAGE_DIRECTORY)

d = model(training_images, training_labels, test_images, test_labels, num_iterations=2000, learning_rate=0.009,
          print_cost=True)

viz_images = (test_images.T).reshape(test_images.shape[1], 256, 256)
visualize_img(viz_images, test_labels.T, d["Y_prediction_test"].T, 0)
