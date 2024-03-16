from linear_reg import LinearRegression # import the local module
import matplotlib.pyplot as plt # library to plot
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
np.random.seed(10) # set seed to generate reproduce same data points

# load dataset
X , y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split data for training.

# Model
regressor = LinearRegression(learning_rate=0.01, n_iterations=1000)

regressor.fit(X_train, y_train) # fitting the model
predictions = regressor.predict(X_test)# making predictions

# model evaluation
mse = regressor.mean_squared_error(y_test, predictions)
accuracy = regressor.r2_score(y_test, predictions)
print(mse)
print(accuracy)

#plot results
predictions_line = regressor.predict(X)
fig = plt.figure(figsize=(8, 5))
r1 = plt.scatter(X_train, y_train, label='train set')
r2 = plt.scatter(X_test, y_test, label='test set')
plt.plot(X, predictions_line, color='black', label="best fit line")
plt.legend()
plt.show()

