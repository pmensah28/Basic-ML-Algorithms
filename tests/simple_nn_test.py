import numpy as np
import matplotlib.pyplot as plt
from simple_nn import SimpleNeuralNetwork

# Generate data to train and test the model.
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
input_layer_size = 2
hidden_layer_size = 10  # Example choice
output_layer_size = 1  # Binary classification
learning_rate = 0.1
n_epochs = 1000  # Number of epochs for training

nn_model = SimpleNeuralNetwork(input_layer_size, hidden_layer_size, output_layer_size, learning_rate)
# nn_model.fit(X_train, Y_train, X_test, Y_test, n_epochs)
# train_predictions = nn_model.predict(X_train)
# test_predictions = nn_model.predict(X_test)
#
# # check for train accuracy and test accuracy
# train_accuracy = nn_model.accuracy(train_predictions, Y_train)
# test_accuracy = nn_model.accuracy(test_predictions, Y_test)
# print("Train accuracy: ", train_accuracy)
# print("Test accuracy: ", test_accuracy)


#     plt.plot(train_loss, label='Train Loss')
#     plt.plot(test_loss, label='Test Loss')
#     plt.legend()
#     plt.title('Training and Test Loss Over Epochs')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.show()
#
#     y_pred_train = self.predict(X_train)
#     train_accuracy = self.accuracy(y_pred_train, Y_train)
#     print("Train accuracy:", train_accuracy)
#
#     y_pred_test = self.predict(X_test)
#     test_accuracy = self.accuracy(y_pred_test, Y_test)
#     print("Test accuracy:", test_accuracy)
#
# nn.train(X_train, Y_train, X_test, Y_test, n_epochs)