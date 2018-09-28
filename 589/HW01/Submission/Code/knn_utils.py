import numpy as np

class KnnUtils(object):
    def __init__(self):
        pass

    def train(self, x, y):
        self.X_train = x
        self.Y_train = y

    def calculate_distance(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        print((X**2).shape)
        X2 = np.sum(X**2, axis=1)
        X2 = np.reshape(X2, (num_test, 1))
        X_train2 = np.reshape(np.sum(self.X_train ** 2, axis=1), (1, num_train))
        sum_term = X2 + X_train2
        pdt_term = -2 * (X @ np.transpose(self.X_train))
        dists = np.sqrt(sum_term + pdt_term)
        return dists

    def predict(self, dists, k):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = self.y_train[(np.argsort(dists[i, :]))[0:k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred
