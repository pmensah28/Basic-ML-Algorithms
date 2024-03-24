import numpy as np
import matplotlib.pyplot as plt

class SimpleNeuralNetwork:
    def __init__(self, input_layer, hidden_layer, output_layer, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.W1, self.W2, self.b1, self.b2 = self.init_params(input_layer, hidden_layer, output_layer)

    def sigmoid(self, z): # the sigmoid function
        return 1 / 1 + np.exp(-z)

    def d_sigmoid(self, z): # derivative of the sigmoid function
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def loss(self, y_pred, Y):
        m = Y.shape[1]
        return -np.sum(Y * np.log(y_pred) + (1 - Y) * np.log(1 - y_pred)) / m


    def init_params(self, input_layer, hidden_layer, output_layer): # Initialize parameter for training the model
        W1 = np.random.randn(hidden_layer, input_layer) * np.sqrt(2/(input_layer + hidden_layer))
        W2 = np.random.randn(output_layer, hidden_layer) * np.sqrt(2 / (hidden_layer + output_layer))
        b1 = np.random.randn(hidden_layer, 1)
        b2 = np.random.randn(output_layer, 1)
        return W1, W2, b1, b2


    def forward_pass(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)
        return A2, Z2, A1, Z1

    def backward_pass(self, X, Y, A2, Z2, A1, Z1):
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.dot(self.W2.T, dZ2) * self.d_sigmoid(Z1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        return dW1, dW2, db1, db2

    def update_params(self, dW1, dW2, db1, db2): # Update parameters
        self.W1 -= self.learning_rate * dW1
        self.W2 -= self.learning_rate * dW2
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2

    # the fit function
    def fit(self, X_train, Y_train, X_test, Y_test, n_epochs):
        train_loss, test_loss = [], []

        for i in range(n_epochs):
            A2, Z2, A1, Z1 = self.forward_pass(X_train)
            dW1, dW2, db1, db2 = self.backward_pass(X_train, Y_train, A2, Z2, A1, Z1)
            self.update_params(dW1, dW2, db1, db2)
            train_loss.append(self.loss(A2, Y_train))
            A2_test, _, _, _ = self.forward_pass(X_test)
            test_loss.append(self.loss(A2_test, Y_test))

    def predict(self, X): # # function for predictions
        predictions, _, _, _ = self.forward_pass(X)
        return predictions > 0.5

    def accuracy(self, y_true, y_pred): # function to calculate accuracy.
        return np.mean(y_true == y_pred)


