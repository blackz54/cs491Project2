import numpy as np
import random as r

def K_Means(X, K):
    numSamps = len(X)
    print("0: ", X)
    # initialize random cluster centers
    r.seed(None)
    takenCenters = [r.randint(0, numSamps - 1) for i in range(0, K)]
    centers = [list(X[takenCenters[i]]) for i in range(0, K)]
    while True:
        print("Printing centers: ")
        print(centers)
        C = [[] for i in range(K)]
        for i in range(0, numSamps):
            index = find_closest_center(X[i], centers)
            C[index].append(list(X[i]))
        print("first center element and length")
        print(C[0])
        print(len(C[0]))
        print(np.sum(C[0]))
        newCenters = [np.around(np.sum(C[i])/max(1, len(C[i])), 3) for i in range(0, K)]
        print("testing here")
        print(centers)
        print(newCenters)
        print(np.array_equal(centers, newCenters))
        if np.array_equal(centers, newCenters):
            print("returning")
            print(C)
            return C
        centers = newCenters


def find_closest_center(dataPoint, centers):
    temp = [distance(dataPoint, centers[i]) for i in range(0, len(centers))]
    minPosition = temp.index(min(temp))
    return minPosition


def distance(x, y):
    x = np.array(x)
    y = np.array(y)
    return np.sqrt(np.sum((x - y) ** 2))

