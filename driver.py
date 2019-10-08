import numpy as np
import nearest_neighbors as nn

x = np.array([2, 2, 3])
y = np.array([4, 4, 5])
val = nn.distance(x, y)

X_train = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
Y_train = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])

X_test = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
Y_test = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])

acc = nn.KNN_test(X_train, Y_train, X_test, Y_test, 2)
k = nn.choose_K(X_train, Y_train, X_test, Y_test)
