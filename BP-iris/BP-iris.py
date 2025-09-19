import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris

def prepare_data(target):
    iris = load_iris()
    X_tr = iris.data[:, 2:]
    labels = iris.target_names
    y = iris.target

    y_tr = []
    for i in range(150):
        y_tr.append(labels[u[i]] == target) 
    y_tr = np.array(y_tr, dtype=int)
    return X_tr, y_tr, ['(1) '+target, '(0) the others']

def sigmoid(x):
    ''' x : numpy array '''
    return 1 / (1 + np.exp(-x))

def loss_mse(y, y_hat):
    loss = 0.0
    for i in range(len(y)):
        err = y_hat[i] - y[i]
        loss += np.dot(err, err)
    return loss / len(y)

def loss_ce(y, y_hat):
    loss = 0.0
    if len(y.shape) == 1 or y.shape[1] == 1:
        for i in range(len(y)):
            loss += -(y[i] * np.log(y_hat[i])
                      + (1-y[i] * np.log((1-y_hat[i])))).sum()
    else:
        for i in range(len(y)):
            loss += -(y[i] * np.log(y_hat[i])).sum()
    return loss / len(y)

class Dense():
    def __init__(self, nIn, nOut, activation='sigmoid', loss='mse'):
        self.nIn = nIn  # 입력의 수
        self.nOut = nOut # 출력의 수
        # 가중치(w)와 바이어스(b)를 He normal 방식으로 초기화
        rnd = np.