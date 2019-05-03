
import numpy as np
import pandas as pd

from tensorflow.keras.datasets import mnist

(X_train, y_train),(X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, -1)
X_test = X_test.reshape(10000, -1)
y_train = pd.get_dummies(y_train).values
y_test = pd.get_dummies(y_test).values

epoch = 10;

input_layer = 784
hidden_layer = 784
output_layer = 10

W1 = np.random.randn(input_layer, hidden_layer)
assert(W1.shape==(784,784))
b1 = np.zeros((1, hidden_layer))
assert(b1.shape==(1,784))
W2 = np.random.randn(hidden_layer, output_layer)
assert(W2.shape==(784,10))
b2 = np.zeros((1, output_layer))
assert(b2.shape==(1,10))

def feedforward(X_train, y_train, W1, b1, W2, b2):
	z1 = np.add(X_train.dot(W1), b1)
	a1 = sigmoid(z1)
	z2 = np.add(a1.dot(W2), b2)
	a2 = sigmoid(z2)
	return z1, a1, z2, a2

def backpropagation(X_train, y_train, W1, b1, W2, b2, z1, a1, z2, a2):
	loss = diff_cost_function(y_train, a2)*diff_sigmoid(z2)
	#loss_W1 = 
	nabla_W1 = np.dot(X_train.T, loss.dot(W2.T)*diff_sigmoid(z1))
	nabla_b1 = np.mean(loss.dot(W2.T)*diff_sigmoid(z1), axis=0).reshape(1, -1)
	nabla_W2 = a1.T.dot(loss)
	nabla_b2 = np.mean(loss, axis=0).reshape(1, -1)
	return nabla_W1, nabla_b1, nabla_W2, nabla_b2

def adjust_params(W1, b1, W2, b2, nabla_W1, nabla_b1, nabla_W2, nabla_b2):
	W1 = W1 + nabla_W1
	b1 = b1 + nabla_b1
	W2 = W2 + nabla_W2
	b2 = b2 + nabla_b2
	return W1, b1, W2, b2

def sigmoid(z):
	return 1/(1+np.exp(z))

def diff_sigmoid(z):
	return sigmoid(z)*(1-sigmoid(z))

def cost_function(y_train, output):
	return np.square(y_train - output)

def diff_cost_function(y_train, output):
	return 2*(y_train - output)

for i in range(epoch):
    z1, a1, z2, a2 = feedforward(X_train, y_train, W1, b1, W2, b2)
    nabla_W1, nabla_b1, nabla_W2, nabla_b2 = backpropagation(X_train, y_train, W1, b1, W2, b2, z1, a1, z2, a2)
    W1, b1, W2, b2 = adjust_params(W1, b1, W2, b2, nabla_W1, nabla_b1, nabla_W2, nabla_b2)
    cost_sum = np.sum(cost_function(y_train, a2))
    print(cost_sum)

final_params = (W1, b1, W2, b2)


#print(y_out.shape)