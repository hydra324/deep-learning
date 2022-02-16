from DL_assigment_Course_2_week_2.opt_utils import load_dataset, initialize_parameters, forward_propagation, \
    compute_cost, backward_propagation, predict, predict_dec, plot_decision_boundary
from DL_assigment_Course_2_week_2.test_cases import update_parameters_with_gd_test_case, random_mini_batches_test_case
import cupy as np
import matplotlib.pyplot as plt

def update_params_with_gd(params, gradients, alpha):
    """
    :param params:
    :param gradients:
    :param alpha:
    :return: params
    """
    L = len(params) // 2

    for l in range(1, L+1):
        params["W" + str(l)] = params["W" + str(l)] - alpha * gradients["dW" + str(l)]
        params["b" + str(l)] = params["b" + str(l)] - alpha * gradients["db" + str(l)]

    return params


# parameters, grads, learning_rate = update_parameters_with_gd_test_case()
# parameters = update_params_with_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = Y.shape[1]
    Z = np.concatenate((X, Y), axis=0)
    np.random.seed(seed)
    np.random.shuffle(Z.T)
    X = Z[:-1, :].reshape(X.shape[0], Y.shape[1])
    Y = Z[-1, :].reshape(1, -1)

    num_split_sections = m//mini_batch_size

    mini_batches = []

    prev_split = 0
    for i in range(1, num_split_sections+1):
        mini_batches.append([X[:, prev_split: i*mini_batch_size], Y[:, prev_split: i*mini_batch_size]])
        prev_split = i*mini_batch_size
    if m != prev_split:
        mini_batches.append([X[:, prev_split:], Y[:, prev_split:]])

    return mini_batches


# X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
#
# print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
# print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
# print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
# print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
# print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
# print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
# print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

def initialize_velocity(params):
    L = len(params)//2
    velocity = {}
    for l in range(1, L+1):
        velocity["W"+str(l)] = np.zeros_like(params["W"+str(l)])
        velocity["b"+str(l)] = np.zeros_like(params["b"+str(l)])
    return velocity

def update_params_with_momentum(params, grads, v, beta, alpha):
    L = len(params)//2

    for l in range(1, L+1):
        v["W"+str(l)] = beta * v["W"+str(l)] + (1-beta) * grads["dW"+str(l)]
        v["b"+str(l)] = beta * v["b"+str(l)] + (1-beta) * grads["db"+str(l)]
        params["W"+str(l)] = params["W"+str(l)] - alpha * v["W"+str(l)]
        params["b"+str(l)] = params["b"+str(l)] - alpha * v["b"+str(l)]
    return params, v

def initialize_adam(params):
    L = len(params) // 2
    v = {}
    s = {}
    for l in range(1, L + 1):
        v["W" + str(l)] = np.zeros_like(params["W" + str(l)])
        v["b" + str(l)] = np.zeros_like(params["b" + str(l)])
        s["W" + str(l)] = np.zeros_like(params["W" + str(l)])
        s["b" + str(l)] = np.zeros_like(params["b" + str(l)])
    return v,s

def update_params_with_adam(params, grads, v, s, t, beta1, beta2, alpha, epsilon):
    L = len(params)//2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L+1):
        v["W" + str(l)] = beta1 * v["W" + str(l)] + (1 - beta1) * grads["dW" + str(l)]
        v["b" + str(l)] = beta1 * v["b" + str(l)] + (1 - beta1) * grads["db" + str(l)]
        v_corrected["W"+str(l)] = v["W" + str(l)] / (1-np.power(beta1, t))
        v_corrected["b"+str(l)] = v["b" + str(l)] / (1-np.power(beta1, t))

        s["W" + str(l)] = beta1 * s["W" + str(l)] + (1 - beta2) * np.square(grads["dW" + str(l)])
        s["b" + str(l)] = beta1 * s["b" + str(l)] + (1 - beta2) * np.square(grads["db" + str(l)])
        s_corrected["W"+str(l)] = s["W" + str(l)] / (1-np.power(beta2, t))
        s_corrected["b"+str(l)] = s["b" + str(l)] / (1-np.power(beta2, t))

        params["W"+str(l)] = params["W"+str(l)] - alpha * v_corrected["W"+str(l)] / (np.sqrt(s_corrected["W"+str(l)]) + epsilon)
        params["b"+str(l)] = params["b"+str(l)] - alpha * v_corrected["b"+str(l)] / (np.sqrt(s_corrected["b"+str(l)]) + epsilon)
    return params, v, s

def model(X, Y, layer_dims, optimizer, mini_batch_size=64, learning_rate=0.007, beta=0.9, beta1=0.9, beta2=0.999, num_epochs=10000, epsilon=1e-8, print_cost=True):
    params = initialize_parameters(layer_dims)
    seed = 10
    t = 0  # initializing the counter required for Adam update

    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(params)
    elif optimizer == "adam":
        v, s = initialize_adam(params)

    costs = []

    for i in range(num_epochs):
        seed += 1
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)

        for mini_batch in mini_batches:
            [mini_batch_X, mini_batch_Y] = mini_batch
            a3, cache = forward_propagation(mini_batch_X, params)
            cost = compute_cost(a3, mini_batch_Y)
            grads = backward_propagation(mini_batch_X, mini_batch_Y, cache)

            if optimizer == "gd":
                params = update_params_with_gd(params, grads, learning_rate)
            elif optimizer == "momentum":
                params, v = update_params_with_momentum(params, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t += 1
                params, v, s = update_params_with_adam(params, grads, v, s, t, beta1, beta2, learning_rate, epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost.get())

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show(block=True)

    return params


train_X, train_Y = load_dataset()
train_X, train_Y = np.array(train_X), np.array(train_Y)

# Mini-batch Gradient descent
# train a 3-layer model
layer_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X,train_Y,layer_dims,optimizer="adam")
predict = predict(train_X, train_Y, parameters)

plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T),train_X, train_Y)