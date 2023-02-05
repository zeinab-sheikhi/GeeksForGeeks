from sklearn.datasets import make_blobs, make_moons, make_circles

from matplotlib import pyplot as plt
from matplotlib import style

# Creating test dataset using make_blobs
style.use("fivethirtyeight")
X, y = make_blobs(n_samples=100, centers=3, cluster_std=1, n_features=2)
plt1 = plt.figure(1)
plt.scatter(X[:, 0], X[:, 1], s=40, color='g')

# Creating test dataset using make_moon
X, y = make_moons(n_samples=1000, noise=0.1)
plt2 = plt.figure(2)
plt.scatter(X[:, 0], X[:, 1], s=40, color='r')


# Creating test dataset using make_circles
X, y = make_circles(n_samples=100, noise=0.02)
plt3 = plt.figure(3)
plt.scatter(X[:, 0], X[:, 1], s=40, color='b')

plt.xlabel('X')
plt.ylabel('Y')
plt1.show()
plt2.show()
plt3.show()

input()
