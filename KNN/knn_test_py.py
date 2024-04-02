import numpy as np
from sklearn.datasets import make_moons
from knn import KnnClassifier
# np.random.seed(0)
import matplotlib.pyplot as plt

X, y = make_moons(n_samples=600, noise=0.55, random_state=0)

# shuffle data
idx = np.random.permutation(X.shape[0])
X, y = X[idx,:], y[idx]

# train/test split
ratio = 0.8
X_train, y_train = X[:int (ratio*X.shape[0])], y[:int (ratio*X.shape[0])].reshape(-1,1)

X_test, y_test = X[int (ratio*X.shape[0]):], y[int (ratio*X.shape[0]):].reshape(-1,1)

print (X_train.shape, X_test.shape , y_train.shape, y_test.shape)
knn = KnnClassifier(k=3)
dists = knn.get_distance(X_train,X_test)
knn.predict_labels(dists, y_train, k=3)
accuracy = []
for i in range(1, 20):
  knn = KnnClassifier(i)
  knn.fit(X_train,y_train)
  accuracy.append(knn.accuracy(X_test, y_test))

print(accuracy)
plt.plot(accuracy, label="test accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Test accuracy vs K")
plt.legend()
plt.show()
#
neighbors = [1,2,3,5,7,11,15,99]
# neighbors = [99,479]

for i,k in enumerate(neighbors):
    model = KnnClassifier(k)
    model.fit(X_train, y_train)
    plt.figure(figsize=(8,6))
    model.plot_decision_boundary(X_test, y_test, model, k=k)