import numpy as np
import clustering as cp
import matplotlib.pyplot as plt

X = np.array([[0,1], [1,2], [2,3], [7,8], [8,9], [9,10], [12,13], [14,15], [15,16]])
Y = np.array([[1, 0], [7, 4], [9, 6], [2, 1], [4, 8], [0, 3], [13, 5], [6, 8], [7, 3], [3, 6], [2, 1], [8, 3], [10, 2], [3, 5], [5, 1], [1, 9], [10, 3], [4, 1], [6, 6], [2, 2]])

K = 2
#C = cp.K_Means(X, K)
C = cp.K_Means(Y, K)
plt.matshow(X)
#plt.scatter(X[0],X[1])
#plt.scatter(C)
#plt.show()
print(C)
print("START K_MEANS_BETTER")

N = cp.K_Means_better(Y,K)
print(N)

#plt.show()
