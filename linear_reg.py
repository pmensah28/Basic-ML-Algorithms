#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 18:57:26 2024

@author: princemensah
"""
import numpy as np

class LinearRegression:
    # class constructor
    def __init__(self, learning_rate = 0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None # self weight to none
        self.bias = None # set bias to none

    def fit(self, X, y):
        self.losses = []  # create an empty list to store losses
        n_samples, n_features = X.shape # define the number of samples and feature

        # initialization of weight and bias
        self.weights = np.zeros(n_features) # Initialize the number of samples to zeros
        self.bias = 0 # set bias to 0 because it's just a single value added to the function

        # implement the gradient descent algorithm
        for _ in range(self.n_iterations):
            predicted = np.dot(X, self.weights) + self.bias
            loss = (1 / n_samples) * np.sum((y - predicted)**2)
            self.losses.append(loss) # add computed loss after every iteration to the list

            # compute gradient w.r.t weight and bias
            grad_w = (-2 / n_samples) * np.dot(X.T, (y - predicted))
            grad_b = (-2 / n_samples) * np.sum(y - predicted)

            # update values for weight and bias
            self.weights = self.weights - self.learning_rate * grad_w
            self.bias = self.bias - self.learning_rate * grad_b

    #implementing the prediction method for unseen data
    def predict(self, X_test):
        prediction = np.dot(X_test, self.weights) + self.bias
        return prediction

    # Implementing the mean square error metric to calculate the error of the model
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    # implementing the r2_score metric to calculate the accuracy of the model
    def r2_score(self, y_true, y_pred):
        correlation_matrix = np.corrcoef(y_true, y_pred)
        correlation = correlation_matrix[0, 1]
        return correlation ** 2