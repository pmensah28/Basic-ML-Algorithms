import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class KnnClassifier:
  def __init__(self, k):
    self.k = k

  def get_distance(self, x_train, x_test):
    # break the dist calculation into parts
    dist_xtrain = np.sum(x_train**2,axis=1,keepdims=True)
    dist_xtest = np.sum(x_test**2,axis=1,keepdims=True).T
    xtrainxtest = 2* np.dot(x_train, x_test.T)
    # print(np.sqrt(dist_xtrain + dist_xtest - xtrainxtest))
    return np.sqrt(dist_xtrain + dist_xtest - xtrainxtest)

  def fit(self, x_train, y_train):
    self.x_train = x_train
    self.y_train = y_train

  def predict_labels(self, dists, y_train, k=1):
    n_test = dists.shape[1]
    y_pred = np.zeros((n_test, 1))
    # predict the label for each example in x_test
    for i in range(n_test):
      # get the closest k examples
      knn_indices = np.argsort(dists[:, i])[:k]
      # Get the labels for the closest k
      knn_labels = y_train[knn_indices].reshape(-1)
      # Use the majority vote to predict the label
      y_pred[i] = Counter(knn_labels).most_common()[0][0]
    # print(y_pred)
    return y_pred

  def predict(self, x_test):
    distances = self.get_distance(self.x_train, x_test)
    predictions = self.predict_labels(distances, self.y_train, self.k)
    return predictions

  def accuracy(self, x_test, y_test):
    y_pred = self.predict(x_test)
    accuracy = np.sum(y_pred == y_test)/ y_test.shape[0]
    return accuracy

  def plot_decision_boundary(self, X, y, model,k):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print(xx.shape)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f"Decision Boundary for k={k}")
    plt.show()
