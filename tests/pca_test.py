from models.pca import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X = iris['data']
y = iris['target']


n_samples, n_features = X.shape

print('Number of samples:', n_samples)
print('Number of features:', n_features)

pca = PCA(n_components = 2)
pca.fit(X)
X_projected = pca.transform(X)
print('Number of components:', pca.components)
print('\nExplained variance:', pca.explained_variance)
print('\nCumulative explained variance:', pca.cum_explained_variance)
print('\nTransformed data shape:', X_projected.shape)

pca.plot1(X) # plot 1: I defined this in the class implementation file.
# plot
plt.figure(figsize=(8, 6))
plt.scatter(X_projected[:, 0], X_projected[:, 1], c=y, edgecolor='none', alpha=0.7, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='label')
plt.show()

