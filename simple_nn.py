#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:13:50 2024

@author: princemensah
"""
import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        self.learning_rate = learning_rate
        self.W1, self.W2, self.b1, self.b2 = self.init_params()

    def init_params(self): # Initializing the parameters
        W1 = np.random.randn(self.hidden_layer, self.input_layer) * np.sqrt(2 / (self.input_layer + self.hidden_layer))
        W2 = np.random.randn(self.output_layer, self.hidden_layer) * np.sqrt(2 / (self.hidden_layer + self.output_layer))
        b1 = np.random.randn(self.hidden_layer, 1)
        b2 = np.random.randn(self.output_layer, 1)
        return W1, W2, b1, b2

    def sigmoid(self, z): # sigmoid function
        return 1 / (1 + np.exp(-z))

    def d_sigmoid(self, z): # derivative of the sigmoid function
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def loss(self, Y, A2): # the cross entropy loss
      m = Y.shape[1]
      return -np.sum((Y * np.log(A2) + (1 - Y) * np.log(1 - A2))) / m

    def forward_pass(self, X): # forward pass
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        return A2, Z2, A1, Z1

    def backward_pass(self, X, Y, A2, Z2, A1, Z1): # backward pass
      m = Y.shape[1]
      dZ2 = A2 - Y
      dW2 = np.dot(dZ2, A1.T) / m
      db2 = np.sum(dZ2, axis=1, keepdims=True) / m
      dZ1 = np.dot(self.W2.T, dZ2) * self.d_sigmoid(Z1)
      dW1 = np.dot(dZ1, X.T) / m
      db1 = np.sum(dZ1, axis=1, keepdims=True) / m
      return dW1, dW2, db1, db2

    def update_params(self, dW1, dW2, db1, db2): # update parameters
      self.W1 -= self.learning_rate * dW1
      self.W2 -= self.learning_rate * dW2
      self.b1 -= self.learning_rate * db1
      self.b2 -= self.learning_rate * db2

    def predict(self, X): # function for prediction
      A2, _, _, _ = self.forward_pass(X)
      return A2 >= 0.5

    def accuracy(self, Y, Y_pred): # function for computing accuracy.
      return np.mean(Y == Y_pred)

    def fit(self, X_train, Y_train, X_test, Y_test, n_epochs): # the fit function
      self.train_loss, self.test_loss = [], []
      for i in range(n_epochs):
        A2, Z2, A1, Z1 = self.forward_pass(X_train)
        dW1, dW2, db1, db2 = self.backward_pass(X_train, Y_train, A2, Z2, A1, Z1)
        self.update_params(dW1, dW2, db1, db2)
        self.train_loss.append(self.loss(Y_train, A2))
        A2_test, _, _, _ = self.forward_pass(X_test)
        self.test_loss.append(self.loss(Y_test, A2_test))
        if i % 1000 == 0:
          self.plot_decision_boundary(X_train, Y_train)

    def plot_decision_boundary(self, X, Y): # function to plot the decision boundaries
      x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
      y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
      h = 0.01
      xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
      Z = self.predict(np.c_[xx.ravel(), yy.ravel()].T)
      Z = Z.reshape(xx.shape)
      plt.figure(figsize=(10, 8))
      plt.contourf(xx, yy, Z, alpha=0.8)
      plt.scatter(X[0, :], X[1, :], c=Y[0, :], edgecolors='k', s=25)
      plt.show()


