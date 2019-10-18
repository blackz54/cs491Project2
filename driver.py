import numpy as np
import matplotlib.pyplot as plt
import nearest_neighbors as nn
import perceptron as pt
import clustering as cl

x = np.array([2, 2, 3])
y = np.array([4, 4, 5])
val = nn.distance(x, y)

X_train = np.array([[1, 5], [2, 6], [2, 7], [3, 7], [3, 8], [4, 8], [5, 1], [5, 9], [6, 2], [7, 2], [7, 3], [8, 3], [8, 4], [9, 5]])
Y_train = np.array([[-1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [1]])

X_test = np.array([[1, 1], [2, 1], [0, 10], [10, 10], [5, 5], [3, 10], [9, 4], [6, 2], [2, 2], [8, 7]])
Y_test = np.array([[1], [-1], [1], [-1], [1], [-1], [1], [-1], [1], [-1]])

acc = nn.KNN_test(X_train, Y_train, X_test, Y_test, 2)
k = nn.choose_K(X_train, Y_train, X_test, Y_test)

'''
print("Problem 2")
X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
X_two = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])
K = 2
centers, C = cl.K_Means(X_two, K)
#N = cl.K_Means_better(X_two ,K)
#print("leaving kmb")
#print(N)
print(centers)
print(C)
print("plotting k_means:")
for i in range(0, len(C)):
    data = np.array(C[i])
    x, y = data.T
    plt.scatter(x, y)
    plt.plot(centers[i][0], centers[i][1], marker='o', markersize=3, color="black")
n = [i for i in range(len(centers))]
for i, txt in enumerate(n):
    plt.annotate(txt, (centers[i][0], centers[i][1]))
plt.title("2-Means")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
'''

print("Problem 3")
#X = np.array([[0, 1], [1, 0], [5, 4], [1, 1], [3, 3], [2, 4], [1, 6]])
#Y = np.array([[1], [1], [0], [1], [0], [0], [0]])
#X = np.array([[-1, -1], [-1, 0], [0, -1], [1, 1]])
#Y = np.array([[-1], [-1], [-1], [1]])

#W = pt.perceptron_train(X, Y)
#test_acc = pt.perceptron_test(X, Y, W[0], W[1])

X = np.array([[-2, 1], [1, 1], [1.5, -0.5], [-2, -1], [-1, -1.5], [2, -2]])
Y = np.array([[1], [1], [1], [-1], [-1], [-1]])
W = pt.perceptron_train(X, Y)
test_acc = pt.perceptron_test(X, Y, W[0], W[1])
print(W[0])
print(W[1])
for i in range(0, len(X)):
    data = np.array(X[i])
    x, y = data.T
    plt.plot(x, y, marker='o', markersize=3, color="black")
varX = np.linspace(-3, 3, 100)
plt.show()

