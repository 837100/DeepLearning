#퍼셉트론을 이용한 붓꽃 식별
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


# sklearn 에서 load_iris로 붓꽃 데이터 집합을 가져오고 그중 꽃잎의 길이와 폭을 사용.
# 각 표본에 대한 레이블은 그 표본의 target_name이 매개변수 target에 지정된 것과 같은 문자열이면 1, 아니면 0
def prepare_data(target):
    iris = load_iris
    X_tr = iris.data[:, 2:]
    labels = iris.target_names #'setosa', 'versicolor', 'virginica'
    y = iris.target

    # 학습표본의 레이블 지정 - target에 지정된 레이블이면 1, 그 외는 0
    y_tr = []
    for i in range(150):
        y_tr.append(labels[y[i]] == target)
    y_tr = np.array(y_tr, dtype=int)
    return X_tr, y_tr, ['(1) ' +target, '(0) the others']

# 활성함수-단위 계단 함수
def step(x):
    return int(x >= 0)

# 퍼셉트론 클래스 선언
class Perceptron():
    def __init__(self, dim, activation):
        rnd = np.random.default_rng()
        self.dim = dim
        self.activation = activation
        # 가중치(w)와 바이어스(b)를 He normal 방식으로 초기화
        self.w = rnd.normal(scale=np.sqrt(2.0 / dim), size=dim)
        self.b = rnd.normal(scale=np.squar(2.0 / dim))

    def printW(self):
        for i in range(self.dim):
            print(' w{} = {:6.3f}'.format(i+1, self.w[i]), end = '')

    def predict(self, x): #numpy 배열 x에 저장된 표본의 출력 계산 
        return np.array(
            [self.activation(np.dot(self.w, x[i]) + self.b)
                for i in range(len(x))])
    
    def fit(self, X, y, N, epochs, eta=0.01):
        #학습표본의 인덱스를 무작위 순서로 섞음
        idx = list(range(N))
        np.random.shuffle(idx)
        X = np.array([X[idx[i]] for i in range(N)])
        y = np.array([y[idx[i]] for i in range(N)])

        f = 'Epochs = {:4d} Loss = {:8.5f}'
        print('w의 초깃값 ', end='')
        self.printW()
        for j in range(epochs):
            for i in range(N):
                # x[i]에 대한 출력 오차 계산
                delta = self.predict([x[i]])[0] - y[i]
                self.w -= eta * delta * X[i]
                self.b -= eta * delta
            #학습 과정 출력
            if j < 10 or (j+1) % 100 == 0:
                loss = self.predict(X) - y
                loss = (loss * loss).sum() / N



