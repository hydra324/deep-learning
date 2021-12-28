# Package imports
import numpy as np
import matplotlib.pyplot as plt
from test_cases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# %matplotlib inline

np.random.seed(1)  # set a seed so that the results are consistent

# load dataset
X, Y = load_planar_dataset()
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

m = X.shape[1]
print('The shape of X is: '+str(X.shape))
print('The shape of Y is: '+str(Y.shape))
print('I ahve m= %d training examples!' % (m))

# let's first try out logistic regression before building a neural network model
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title('Logistic Regression')

# print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# As you can see, logistric regression did not do well with our data
# let's now build a neural network model with one hidden layer

def layer_sizes(X, Y):
    n_x = X.shape[0] # number of features in input, here its 2
    n_h = 4 # number of hidden layer units
    n_y = Y.shape[0] # number of outputs, here its 1
    return (n_x,n_h,n_y)

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)* 0.01
    b2 = np.zeros((n_y,1))
    
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    return parameters

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1,X)+b1
    A1 = np.tanh(Z1)
    
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
    return cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    cost = (-1/m)* np.sum(Y*np.log(A2) + (1-Y)*np.log(1-A2))
    cost = np.squeeze(cost)
    return cost

def back_propagation(parameters, cache, X,Y):
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    m = Y.shape[1]
    
    dZ2 = A2-Y
    dW2 = (1/m)* np.dot(dZ2,A1.T)
    db2 = (1/m)* np.sum(dZ2,axis=1,keepdims=True)
    g1prime = 1-np.power(A1,2)
    dZ1 = np.multiply(np.dot(W2.T,dZ2),g1prime)
    dW1 = (1/m)* np.dot(dZ1,X.T)
    db1 = (1/m)* np.sum(dZ1,axis=1,keepdims=True)
    
    grads = {'dW1':dW1,'db1':db1,'dW2':dW2,'db2':db2}
    return grads

def update_parameters(parameters,grads,learning_rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    return parameters

def nn_model(X,Y,n_h,num_iterations=10000,print_cost=False):
    np.random.seed(3)
    n_x,_,n_y = layer_sizes(X,Y)
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    costs = []
    for i in range(0,num_iterations):
        cache = forward_propagation(X,parameters)
        cost = compute_cost(cache['A2'],Y,parameters)
        
        grads = back_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads)
        if print_cost and i%1000 ==0:
            print("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    return parameters,costs

def predict(parameters, X):
    cache = forward_propagation(X,parameters)
    predictions = (cache['A2'] > 0.5)
    return predictions

parameters,costs = nn_model(X,Y,n_h=4,num_iterations=10000,print_cost=True)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

# Print accuracy
predictions = predict(parameters, X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

# Tuning hidden layer size
# This may take about 2 minutes to run

plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i+1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters,costs = nn_model(X, Y, n_h, num_iterations = 5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    
    
# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_circles"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);