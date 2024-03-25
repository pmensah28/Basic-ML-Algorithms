from numpy.linalg import eig
import numpy as np
import matplotlib.pyplot as plt
class PCA:
  def __init__(self, n_components):
    self.components = n_components

  def fit(self, X):
    self.mean = np.sum(X, axis = 0) / X.shape[0]
    self.std = (np.sum((X - self.mean)**2, axis=0) / (X.shape[0] - 1))**.5
    self.standardize = (X - self.mean) / self.std

    # Eigendecomposition of the covariance matrix
    covariance_mat = self.standardize.T @ self.standardize / (X.shape[0] - 1)
    eigen_values, eigen_vectors = eig(covariance_mat)

    # rank the eigenvalues and the associated eigenvectors in descending order.
    idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
    eigen_values_sorted = eigen_values[idx]
    self.eigen_vectors_sorted = eigen_vectors.T[:,idx]

    self.explained_variance = [(i / sum(eigen_values))*100 for i in eigen_values_sorted]
    self.explained_variance = np.round(self.explained_variance, 2)
    self.cum_explained_variance = np.cumsum(self.explained_variance)

    return self

    # Transform data
  def transform(self, X):
    X_std = (X - self.mean) / self.std
    P = self.eigen_vectors_sorted[:self.components, :] # Projection matrix
    X_proj = X_std.dot(P.T)

    return X_proj

  def plot1(self, X):
    plt.plot(np.arange(1,X.shape[1]+1), self.cum_explained_variance, '-o')
    plt.xticks(np.arange(1,X.shape[1]+1))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance');
    plt.grid()
    plt.show()


