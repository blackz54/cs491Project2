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
        newCenters = [around(np.mean(C[i], axis=0), 2) if len(C[i]) > 0 else 0 for i in range(0, K)]
        #newCenters = [around(mean(C[i]), 2) if len(C[i]) > 0 else np.array([0]*len(X[0])) for i in range(0, K)]
        if array_equal(centers, newCenters):
            return np.array(centers)
        centers = newCenters


def mean(C):
    value = 0
    groups = []
    if C is None:
        return None
    for i in range(0, len(C[0])):
        for j in range(0, len(C)):
            value += C[j][i]
        value = float(value)/len(C)
        groups.append(value)
    return groups


def around(X, d):
    return [round(float(X[i]), 2) for i in range(len(X))]


def array_equal(X, Y):
    for i in range(0, len(X)):
        if not tuple(X[i]) == tuple(Y[i]):
            return False
    return True


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


def K_Means_better(X, K):
    centers = []
    commonCenters = []
    centers_count = []
    max_runCount = 1000
    # run K_Means max_runCount times and count centers generated
    for i in range(0, max_runCount):
        centers.append(tuple(K_Means(X, K)))
        if list(centers[i]) in list(commonCenters):
            check = 0
            while not centers[i] == commonCenters[check]:
                check += 1
            centers_count[check] += 1
        else:
            commonCenters.append(centers[i])
            centers_count.append(1)
    mode_of_centers = Get_Ind_of_Max(centers_count)
    return np.array(commonCenters[mode_of_centers])


def Get_Ind_of_Max(M):
    maxInd = 0
    for i in range(0, len(M)):
        if M[maxInd] < M[i]:
            maxInd = i
    return maxInd
