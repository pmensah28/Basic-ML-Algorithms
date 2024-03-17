#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:15:26 2024

@author: princemensah
"""
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, method='GD', batch_size=None, momentum=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.method = method
        self.batch_size = batch_size
        self.momentum = momentum
        self.weights = None
        self.velocity = None
        self.loss_history = []  # Store the loss here at each iteration

    def add_ones(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def grad_mse_loss(self, X, y_true, y_pred):
        return -2 * np.dot(X.T, (y_true - y_pred)) / len(y_true)

    def update_weights(self, gradient):
        if self.momentum:
            if self.velocity is None:
                self.velocity = np.zeros_like(gradient)
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            self.weights += self.velocity
        else:
            self.weights -= self.learning_rate * gradient

    def fit(self, X, y):
        X = self.add_ones(X)
        self.weights = np.zeros(X.shape[1])
        self.velocity = None if self.momentum is None else np.zeros(X.shape[1])

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.weights)
            loss = self.mse_loss(y, y_pred)
            self.loss_history.append(loss)  # Append the current loss to the history
            gradient = self.grad_mse_loss(X, y, y_pred)
            self.update_weights(gradient)

    def predict(self, X):
        X = self.add_ones(X)
        return np.dot(X, self.weights)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title('Loss over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('MSE Loss')
        plt.show()


class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000, method='GD', batch_size=None, momentum=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.method = method
        self.batch_size = batch_size
        self.momentum = momentum
        self.weights = None
        self.velocity = None
        self.loss_history = []

    def add_ones(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def grad_cross_entropy(self, X, y_true, y_pred):
        return np.dot(X.T, (y_pred - y_true)) / X.shape[0]

    def update_weights(self, gradient):
        if self.momentum:
            if self.velocity is None:
                self.velocity = np.zeros_like(gradient)
            self.velocity = self.momentum * self.velocity - self.learning_rate * gradient
            self.weights += self.velocity
        else:
            self.weights -= self.learning_rate * gradient

    def fit(self, X, y):
        X = self.add_ones(X)
        self.weights = np.zeros(X.shape[1])
        self.velocity = None if self.momentum is None else np.zeros(X.shape[1])

        for _ in range(self.iterations):
            y_pred = self.sigmoid(np.dot(X, self.weights))
            loss = self.cross_entropy(y, y_pred)
            self.loss_history.append(loss)
            gradient = self.grad_cross_entropy(X, y, y_pred)
            self.update_weights(gradient)

    def predict_proba(self, X):
        X = self.add_ones(X)
        return self.sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.title('Loss over iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Cross-entropy Loss')
        plt.show()