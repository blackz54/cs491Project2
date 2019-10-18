import numpy as np


def KNN_test(X_train, Y_train, X_test, Y_test, K):
    predict = []
    for i in range(0, len(X_test)):
        predict.append(KNN_predict(X_train, Y_train, X_test[i], Y_test[i], K))
    Y_test = np.array(Y_test).flatten()
    correct = 0
    total = 0
    for i in range(0, len(Y_test)):
        if Y_test[i] == predict[i]:
            correct += 1
            total += 1
        else:
            total += 1
    acc = correct/total
    return acc

def distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def KNN_predict(X_train, Y_train, X_test, Y_test, K):
    S = [[distance(X_train[i], X_test), i] for i in range(0, len(X_train))]
    S = sorted(S, key=lambda x: x[0])
    prediction = 0
    for i in range(0, K):
        prediction += Y_train[S[i][1]]
    if prediction > 0:
        return 1
    else:
        return -1


def choose_K(X_train, Y_train, X_val, Y_val):
    best_K = 1
    best_acc = 0
    for i in range(0, len(X_train)):
        acc = KNN_test(X_train, Y_train, X_val, Y_val, i)
        if acc > best_acc:
            best_acc = acc
            best_K = i
    print(best_K)
    print(best_acc)
    return best_K

