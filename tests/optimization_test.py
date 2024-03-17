#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 21:21:26 2024

@author: princemensah

"""

from optimization import LinearRegression
from optimization import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
np.random.seed(10)

def generate_data(n= 1000):
  np.random.seed(0)
  x = np.linspace(-5.0, 5.0, n).reshape(-1,1)
  y= (29 * x + 30 * np.random.rand(n,1)).squeeze()
  x = np.hstack((np.ones_like(x), x))
  return x,y

# Usage Example
np.random.seed(0)
x, y = generate_data()
x_train, y_train, x_test, y_test = split_data(x, y)

model = LinearRegression(learning_rate=0.01, iterations=1000, method='GD', momentum=None)
model.fit(x_train, y_train)
model.plot_loss()

predictions = model.predict(x_test)
print(f"\n\nTest MSE: {model.mse_loss(y_test, predictions)}")


# Example usage
X_class, y_class = make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
X_train, y_train, X_test, y_test = split_data_1(X_class, y_class)

model = LogisticRegression(learning_rate=0.01, iterations=1000, method='GD', momentum=None)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
model.plot_loss()

# Evaluate the model
print(f"\n\nAccuracy: {accuracy_score(y_test, predictions)}")
