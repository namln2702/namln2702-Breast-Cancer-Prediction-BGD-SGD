import numpy as np

def sigmoid(z):
    return 1 /(1 + np.exp(-z))

class SGD:
    def __init__(self, lr = 0.0001, n_iter = 100, batch_size = 50):
        self.lr = lr
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.w = None
        self.b = None

    def gradient_descent(self, X, y):
        n_samples = X.shape[0]
        z = np.dot(self.w, X.T) + self.b
        a = sigmoid(z)

        gradient_weight = (1 / n_samples) * np.dot(a - y, X)
        gradient_bias = (1 / n_samples) * np.sum(a - y)

        self.w -= self.lr * gradient_weight
        self.b -= self.lr * gradient_bias


    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((1,n_features)) # (n, )
        self.b = 0

        for _ in range(self.n_iter):
            for i in range(0, n_samples, self.batch_size):
                X_batch = X[i: i + self.batch_size]
                y_batch = y[:, i : i + self.batch_size]
                self.gradient_descent(X_batch, y_batch)

    def predict(self, X):
        h = np.dot(self.w, X.T) + self.b
        pre = sigmoid(h)

        pred = [1 if i >= 0.5 else 0 for i in pre[0]]
        return pred