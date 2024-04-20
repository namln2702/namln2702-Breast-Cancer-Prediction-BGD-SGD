import numpy as np

def sigmoid(z):
    return 1 /(1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, lr = 0.0001, n_iter = 100):
        self.lr = lr
        self.n_iter = n_iter
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples , n_features = X.shape
        self.w = np.zeros((1,n_features)) # (n, )
        self.b = 0

        for _ in range(self.n_iter):
            z = np.dot(self.w, X.T) + self.b
            a = sigmoid(z)

            dw = ( 1/ n_samples) * np.dot((a - y), X)
            db = ( 1/ n_samples) * np.sum(a - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        h = np.dot(self.w, X.T) + self.b
        pre = sigmoid(h)

        pred = [1 if i >= 0.5 else 0 for i in pre[0]]
        return pred