#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:54:34 2024

@author: princemensah
"""

import numpy as np
import matplotlib.pyplot as plt

from simple_nn import SimpleNeuralNetwork
np.random.seed(10)

# Generate data
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4, 2)
class_0_b = var * np.random.randn(n//4, 2) + (2, 2)
class_1_a = var * np.random.randn(n//4, 2) + (0, 2)
class_1_b = var * np.random.randn(n//4, 2) + (2, 0)

X = np.concatenate([class_0_a, class_0_b, class_1_a, class_1_b], axis=0)
Y = np.concatenate([np.zeros((n//2, 1)), np.ones((n//2, 1))])

# Shuffle the data
rand_perm = np.random.permutation(n)
X = X[rand_perm, :]
Y = Y[rand_perm, :]

# Transpose X and Y to match the neural network's expected input shape
X = X.T
Y = Y.T

# Split the data into training and test sets
ratio = 0.8
X_train = X[:, :int(n * ratio)]
Y_train = Y[:, :int(n * ratio)]
X_test = X[:, int(n * ratio):]
Y_test = Y[:, int(n * ratio):]

# Initialize and train the neural network
h0 = 2 #Input layer
h1 = 10 # hidden layer
h2 = 1 #output layer
learning_rate = 0.1
n_epochs = 1000



nn_model = SimpleNeuralNetwork(h0, h1, h2, learning_rate=0.1) # model
nn_model.fit(X_train, Y_train, X_test, Y_test, n_epochs=10000)
train_preds = nn_model.predict(X_train)
test_preds = nn_model.predict(X_test)
train_accuracy = nn_model.accuracy(Y_train, train_preds)
test_accuracy = nn_model.accuracy(Y_test, test_preds)

# checking for training and test accuracy
print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

# plotting the training loss and testing loss
plt.plot(nn_model.train_loss, label='Train Loss')
plt.plot(nn_model.test_loss, label='Test Loss')
plt.legend()
plt.title('Training and Test Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


