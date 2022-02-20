import tensorflow as tf
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework.ops import EagerTensor

train_dataset = h5py.File('./train_signs.h5', "r")
test_dataset = h5py.File('./test_signs.h5', "r")

x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])

unique_labels = set()
for element in y_train:
    unique_labels.add(element.numpy())
print(unique_labels)

images_iter = iter(x_train)
labels_iter = iter(y_train)
plt.figure(figsize=(10,10))
for i in range(25):
    ax = plt.subplot(5,5,i+1)
    plt.imshow(next(images_iter).numpy().astype("uint8"))
    plt.title(next(labels_iter).numpy().astype("uint8"))
    plt.axis("off")
plt.show(block=True)

def normalize(image):
    """
    Transofrm an image into a tensor of shape (64* 64 * 3, )
    and normalize its components.
    :param image: A Tensor containing image
    :return: Transformed tensor
    """
    image = tf.cast(image, tf.float32) / 255.
    image = tf.reshape(image, [-1, ])
    return image


new_train = x_train.map(normalize)
new_test = x_test.map(normalize)

print(x_train.element_spec)
print(new_train.element_spec)

def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name='X')
    W = tf.Variable(np.random.randn(4,3), name='W')
    b = tf.Variable(np.random.randn(4,1), name='b')
    Y = tf.add(tf.matmul(W, X), b)

    return Y

def sigmoid(z):
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a

def one_hot_matrix(label, depth=6):
    one_hot = tf.reshape(tf.one_hot(label, depth, axis=0), (depth,))
    return one_hot


new_y_train = y_train.map(one_hot_matrix)
new_y_test = y_test.map(one_hot_matrix)

def initialize_parameters():
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    W1 = tf.Variable(initializer(shape=(25,12288)))
    b1 = tf.Variable(initializer(shape=(25,1)))
    W2 = tf.Variable(initializer(shape=(12,25)))
    b2 = tf.Variable(initializer(shape=(12,1)))
    W3 = tf.Variable(initializer(shape=(6,12)))
    b3 = tf.Variable(initializer(shape=(6,1)))

    params = {'W1':W1,'b1':b1,'W2':W2,'b2':b2,'W3':W3,'b3':b3}

    return params

def forward_prop(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    W3 = params['W3']
    b3 = params['b3']

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.keras.activations.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.keras.activations.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

def compute_cost(logits, labels):
    cost = tf.reduce_mean(tf.keras.metrics.categorical_crossentropy(tf.transpose(labels), tf.transpose(logits), from_logits=True))
    return cost

def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    costs = []
    train_acc = []
    test_acc = []

    parameters = initialize_parameters()
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    parameters
    m = dataset.cardinality().numpy()

    minibatches = dataset.batch(minibatch_size).prefetch(8)
    test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)

    for epoch in range(num_epochs):
        epoch_cost = 0.
        train_accuracy.reset_states()

        for (minibatch_X, minibatch_Y) in minibatches:
            with tf.GradientTape() as tape:
                Z3 = forward_prop(tf.transpose(minibatch_X), parameters)
                minibatch_cost = compute_cost(Z3, tf.transpose(minibatch_Y))
            train_accuracy.update_state(tf.transpose(Z3), minibatch_Y)

            trainable_variables = [W1, b1, W2, b2, W3, b3]
            grads = tape.gradient(minibatch_cost, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))
            epoch_cost += minibatch_cost
        epoch_cost /= m

        if print_cost == True and epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            print("Train accuracy:", train_accuracy.result())

            for (minibatch_X,minibatch_Y) in test_minibatches:
                Z3 = forward_prop(tf.transpose(minibatch_X), parameters)
                test_accuracy.update_state(tf.transpose(Z3), minibatch_Y)
            print("Test accuracy:", test_accuracy.result())

            costs.append(epoch_cost)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
            test_accuracy.reset_states()

    return parameters, costs, train_acc, test_acc


parameters, costs, train_acc, test_acc = model(new_train, new_y_train, new_test, new_y_test, num_epochs=100)

# Plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate ="+str(0.0001))
plt.show(block=True)

# Plot the train accuracy
plt.plot(np.squeeze(train_acc))
plt.ylabel('Train Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show(block=True)

# Plot the test accuracy
plt.plot(np.squeeze(test_acc))
plt.ylabel('Test Accuracy')
plt.xlabel('iterations (per fives)')
plt.title("Learning rate =" + str(0.0001))
plt.show(block=True)