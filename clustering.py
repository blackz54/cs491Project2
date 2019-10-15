import numpy as np
import random as r


def K_Means(X, K):
    if K <= 0:
        return X
    numSamps = len(X)
    # initialize random cluster centers
    r.seed(None)
    takenCenters = initialize_unique_centers(numSamps, K)
    centers = [list(X[takenCenters[i]]) for i in range(0, K)]
    while True:
        C = [[] for i in range(K)]
        for i in range(0, numSamps):
            index = find_closest_center(X[i], centers)
            C[index].append(list(X[i]))
        newCenters = [np.around(np.mean(C[i], axis=0), 3) for i in range(0, K)]
        if np.array_equal(centers, newCenters):
            print("returning")
            print(centers)
            print(C)
            return C
        centers = newCenters


def initialize_unique_centers(numSamps, K):
    takenCenters = [r.randint(0, numSamps - 1) for i in range(0, K)]
    for x in takenCenters:
        if takenCenters.count(x) > 1:
            return initialize_unique_centers(numSamps, K)
    return takenCenters


def find_closest_center(dataPoint, centers):
    temp = [distance(dataPoint, centers[i]) for i in range(0, len(centers))]
    minPosition = temp.index(min(temp))
    return minPosition


def distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))
