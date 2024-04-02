from Linear_Regression.linear_reg import LinearRegression # import the local module
import matplotlib.pyplot as plt # library to plot
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
np.random.seed(10) # set seed to generate reproduce same data points

# load dataset
X , y = datasets.make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # split data for training.

# Model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)

model.fit(X_train, y_train) # fitting the model
predictions = model.predict(X_test)# making predictions

# model evaluation
mse = model.mean_squared_error(y_test, predictions)
accuracy = model.r2_score(y_test, predictions)
print('MSE: ', mse)
print('Accuracy: ', accuracy)

# Training error plot
n_points = len(model.losses)
training, = plt.plot(range(n_points), model.losses, label="Training Error", color='blue')
plt.legend(handles=[training])
plt.title("Error Plot")
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.show()

#plot results
predictions_line = model.predict(X)
cmap = plt.get_cmap('Blues') #others: viridis, magma, plasma, cividis, inferno
fig = plt.figure(figsize=(8, 5))
r1 = plt.scatter(X_train, y_train, label='train sample', color=cmap(0.5), s=12)
r2 = plt.scatter(X_test, y_test, label='test sample', color=cmap(0.9), s=12)
plt.plot(X, predictions_line, color='black', label="line of best fit", linewidth=2)
plt.legend(loc='lower right')
plt.show()

