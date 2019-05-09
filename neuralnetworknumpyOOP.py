import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

class NeuralNetwork(object):
	"""
	Arguments:

	input - input should be (no. of instances, columns/features)
	output - output should be (no. of instances, columns/features)
	hiddensize - must be a list
		- specify the size and the number of hidden layers
		- default: None, which means no hidden layer.
		example: 
			hiddenSize = [100, 50, 30] means:
			1st hidden layer has 100 neurons, 
			2nd hidden layer has 50 neurons and
			3rd hidden layer has 30 neurons.
			Number of hidden layers is 3.

	"""

	def __init__(self, input_, output_, hiddenSize=None):
		super(NeuralNetwork, self).__init__()
		self.inputSize = input_.shape[1]
		self.outputSize = output_.shape[1]
		self.hiddenLen = len(hiddenSize)

		self.weights = []
		self.biases = []

		if hiddenSize is None:
			self.weights.append(np.random.randn(self.inputSize, self.outputSize))
			self.biases.append(np.zeros((1, self.outputSize)))
		else:
			for i in range(len(hiddenSize)): # [30] [50, 50, 50, 25, 12] [2]
				if i == 0: # if input X first hidden layer
					self.weights.append(np.random.randn(self.inputSize, hiddenSize[i]))
					self.biases.append(np.zeros((1, hiddenSize[i])))
				elif i == len(hiddenSize) - 1: #if last hidden layer X output layer
					self.weights.append(np.random.randn(hiddenSize[i-1], hiddenSize[i]))
					self.biases.append(np.zeros((1, hiddenSize[i])))
					self.weights.append(np.random.randn(hiddenSize[-1], self.outputSize))
					self.biases.append(np.zeros((1, self.outputSize)))
					break
				else: #if nth hidden layer X nth hidden layer (n is not equal to first or last)
					self.weights.append(np.random.randn(hiddenSize[i-1], hiddenSize[i]))
					self.biases.append(np.zeros((1, hiddenSize[i])))

		self.pre_activations = [0 for i in range(len(self.weights))]
		self.activations = [0 for i in range(len(self.biases))]
		self.nabla_w = [0 for i in range(len(self.weights))]
		self.nabla_b = [0 for i in range(len(self.biases))]

	def checkParameters(self):
		print("Number of Weights: {}".format(len(self.weights)))
		print("Number of Biases: {}".format(len(self.biases)))
		print("Number of Pre_activations: {}".format(len(self.pre_activations)))
		print("Number of Activations: {}".format(len(self.activations)))

		for x, w in enumerate(self.weights):
			print("Weight {} shape : {}".format(x+1, w.shape))

		for x, b in enumerate(self.biases):
			print("Bias {} shape : {}".format(x+1, b.shape))

		for x, p in enumerate(self.pre_activations):
			print("Pre_activation {} shape : {}".format(x+1, p.shape))

		for x, a in enumerate(self.activations):
			print("Activation {} shape : {}".format(x+1, a.shape))

	def fit(self, input_, output_, epochs=10, lr=0.0001):

		for epoch in range(epochs):
			print("Epoch {}".format(epoch+1))
			self.feedforward(input_)
			self.backpropagation(input_,output_)
			self.update_params(lr)
			self.validate_score(output_)
			self.accuracy_metrics(output_)

	def feedforward(self, input_):
		activations = input_
		for i in range(len(self.weights)): # or self.biases
			self.pre_activations[i] = np.add(activations.dot(self.weights[i]), self.biases[i])
			activations = self.sigmoid(self.pre_activations[i])
			self.activations[i] = activations

	def backpropagation(self, input_, output_):
		lossFunction = self.diff_cost_function(output_, self.activations[-1])*self.diff_sigmoid(self.pre_activations[-1])
		loss = lossFunction

		#print(type(lossFunction)) # for debugging

		for i in range(len(self.weights)-1,-1,-1): # backward range: 4, 3, 2 ,1 ,0
			if i==0:
				self.nabla_w[i] = np.dot(input_.T, loss )
				self.nabla_b[i] = np.mean(loss, axis=0).reshape(1,-1)
				break
			#print(type(self.activations[i-1])) # for debugging
			self.nabla_w[i] = np.dot(self.activations[i-1].T , loss)
			self.nabla_b[i] = np.mean(loss, axis=0).reshape(1,-1)
			loss = (loss.dot(self.weights[i].T))*(self.diff_sigmoid(self.pre_activations[i-1]))

	def update_params(self, lr):
		for i in range(len(self.weights)):
			self.weights[i] = self.weights[i] - lr*self.nabla_w[i]
			self.biases[i] = self.biases[i] - lr*self.nabla_b[i]

	def predict(self, input_, output_):
		pre_activations_pred = [0 for i in range(len(self.biases))]
		activations_pred = [0 for i in range(len(self.biases))]
		activations = input_
		for i in range(len(self.weights)): # or self.biases
			pre_activations_pred[i] = np.add(activations.dot(self.weights[i]), self.biases[i])
			activations = self.sigmoid(pre_activations_pred[i])
			activations_pred[i] = activations

		return activations_pred[-1]

	def validate_score(self, output_):
		mse = np.mean(np.sum(self.cost_function(output_, self.activations[-1]), axis=1))
		print("Mean Squared Error: {}".format(mse))

	def accuracy_metrics(self, output_, threshold=0.7):
		self.correct = 0
		self.wrong = 0
		for i in range(len(output_)):
			y = np.array([1 if max(self.activations[-1][i])==j else 0 for j in self.activations[-1][i]])
			print(y.shape)
			print(output_.shape)
			y_num = np.where(y==1)[0][0]
			true_y = np.where(output_[i]==1)[0][0]
			if y_num == true_y:
				self.correct += 1
			else:
				self.wrong += 1
			print(self.correct)
			print(self.wrong)
		print("Accuracy: {:2f} %".format(self.correct/len(output_)*100))

	def sigmoid(self, z):
		return 1/(1+np.exp(z))

	def diff_sigmoid(self, z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def cost_function(self, y_train, output_):
		return 0.5*np.square(y_train - output_)

	def diff_cost_function(self, y_train, output_):
		return (y_train - output_)


def extract_data():
	(X_train, y_train),(X_test, y_test) = mnist.load_data()

	X_train = X_train.reshape(60000, -1) / 255
	X_test = X_test.reshape(10000, -1) / 255
	y_train = pd.get_dummies(y_train).values
	y_test = pd.get_dummies(y_test).values

	return X_train, y_train, X_test, y_test


if __name__ == '__main__':
	X_train, y_train, X_test, y_test = extract_data()

	h_size = X_train.shape[1]

	nn = NeuralNetwork(X_train, y_train, hiddenSize=[h_size, int(h_size/2), int(h_size/2), int(h_size/3), int(h_size/4)])
	nn.fit(X_train, y_train, epochs=3)
	y_pred = nn.predict(X_test, y_test)

	print(y_pred.shape)

	#nn.checkParameters()

