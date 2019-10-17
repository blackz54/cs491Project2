import numpy as np
import clustering as cp

X = np.array([[0], [1], [2], [7], [8], [9], [12], [14], [15]])
K = 3
C = cp.K_Means(X, K)
print(C)
print("START K_MEANS_BETTER")

N = cp.K_Means_better(X,K)
print(N)
