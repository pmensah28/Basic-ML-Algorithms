
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


iris = load_iris()
X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
    )
df["label"] = iris.target

df

df.info()

"""Let's plot our data and see how it's look like"""

column_names = iris.feature_names

plt.figure(figsize=(16,4))
plt.subplot(1, 3, 1)
plt.title(f"{column_names[0]} vs {column_names[1]}")
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.subplot(1, 3, 2)
plt.title(f"{column_names[1]} vs {column_names[2]}")
plt.scatter(X[:, 1], X[:, 3], c=y)
plt.subplot(1, 3, 3)
plt.title(f"{column_names[2]} vs {column_names[3]}")
plt.scatter(X[:, 2], X[:, 3], c=y)
plt.show()

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Display the plot
plt.show()


def mean(X): # np.mean(X, axis = 0)

  # Your code here
  mean = np.sum(X, axis = 0) / X.shape[0]

  return mean

def std(X): # np.std(X, axis = 0)

  # Your code here
  std = (np.sum((X - mean(X))**2, axis=0) / (X.shape[0] - 1))**.5

  return std

def Standardize_data(X):

  # Your code here
  X_std = (X - mean(X)) / std(X)
  return X_std

X_std = Standardize_data(X)

assert (np.round(mean(X_std)) == np.array([0., 0., 0., 0.])).all(), "Your mean computation is incorrect"
assert (np.round(std(X_std)) == np.array([1., 1., 1., 1.])).all(), "Your std computation is incorrect"


def covariance(X):

  ## Your code here
  cov = X.T@ X / (X.shape[0] -1)

  return cov

Cov_mat = covariance(X_std)
Cov_mat


from numpy.linalg import eig

# Your code here
eigen_values, eigen_vectors = eig(Cov_mat)  # return eigen values and eigen vectors

print(eigen_values)
print(eigen_vectors)

"""*   rank the eigenvalues and their associated eigenvectors in decreasing order"""

print(eigen_values)
idx = np.array([np.abs(i) for i in eigen_values]).argsort()[::-1]
print(idx)

print("---------------------------------------------------")

eigen_values_sorted = eigen_values[idx]
eigen_vectors_sorted = eigen_vectors.T[:,idx]

print(eigen_vectors_sorted)


explained_variance = [(i / sum(eigen_values))*100 for i in eigen_values_sorted]
explained_variance = np.round(explained_variance, 2)
cum_explained_variance = np.cumsum(explained_variance)

print('Explained variance: {}'.format(explained_variance))
print('Cumulative explained variance: {}'.format(cum_explained_variance))

plt.plot(np.arange(1,X.shape[1]+1), cum_explained_variance, '-o')
plt.xticks(np.arange(1,X.shape[1]+1))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance');
plt.grid()
plt.show()

"""#### Project our data onto the subspace"""

# Get our projection matrix
c = 2
P = eigen_vectors_sorted[:c, :] # Projection matrix


X_proj = X_std.dot(P.T)
X_proj.shape

plt.title(f"PC1 vs PC2")
plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y)
plt.xlabel('PC1'); plt.xticks([])
plt.ylabel('PC2'); plt.yticks([])
plt.show()