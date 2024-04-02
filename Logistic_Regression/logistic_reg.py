import numpy as np

# building the logistic regression class
class LogisticRegression:
    def __init__(self, learning_rate = 0.001, n_iterations=1000): # class constructor
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None # define weight parameter to None
        self.bias = None # define the bias parameter to None

    def sigmoid(self, x): # sigmoid function method
        return 1 / (1 + np.exp(-x))

    # Negative loglikelihood or the cross-entropy function
    def cross_entropy(self, x, y_true):
        y_pred = self.sigmoid(x)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss


    def fit(self, X, y): # fit method
        self.train_losses = []
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features) # initialize weight to 0
        self.bias = 0 # initialize bias to 0

        for _ in range(self.n_iterations):
            linear_func = np.dot(X, self.weights) + self.bias # built linear model to be used by the logistic function
            predicted = self.sigmoid(linear_func)
            train_losses = self.cross_entropy(y, predicted)
            self.train_losses.append((train_losses))

            # compute gradient of the cross-entropy function w.r.t weight and bias
            grad_w = -(1 / n_samples) * np.dot(X.T, (y - predicted))
            grad_b = -(1 / n_samples) * np.sum(y - predicted)

            # update parameters
            self.weights = self.weights - self.learning_rate * grad_w
            self.bias = self.bias - self.learning_rate * grad_b

    def predict(self, X_test):
        linear_function = np.dot(X_test, self.weights) + self.bias
        predictions = self.sigmoid(linear_function)
        predictions_class = [1 if i >= 0.5 else 0 for i in predictions]
        return np.array(predictions_class)

    def accuracy(self, y_true, y_pred):
        accuracy = (1/ len(y_pred)) * np.sum(y_true == y_pred)
        return accuracy



