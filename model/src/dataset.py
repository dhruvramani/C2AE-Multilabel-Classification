import arff
import numpy as np

class DataSet(object):
    def __init__(self, config):
        self.config = config
        self.train, self.test, self.validation = None, None, None
        self.path = self.config.dataset_path

    def get_data(self, path, noise=False):
        data = np.load(path)
        if noise == True :
            data = data + np.random.normal(0, 0.001, data.shape)
        return data

    def get_train(self):
        if self.train == None:
            X = self.get_data(self.config.train_path + "-features.pkl", True)
            Y = self.get_data(self.config.train_path + "-labels.pkl")
            length = X.shape[0]
            X, Y = X[0 : int(0.8 * length) , :], Y[0 : int(0.8 * length), :]
            self.train = X, Y
        else :
            X, Y = self.train
        return X, Y

    def get_validation(self):
        if self.validation == None:
            X = self.get_data(self.config.train_path + "-features.pkl")
            Y = self.get_data(self.config.train_path + "-labels.pkl")
            length = X.shape[0]
            X, Y = X[0 : int(0.2 * length) , :], Y[0 : int(0.2 * length), :]
            self.validation = X, Y
        else :
            X, Y = self.validation
        return X, Y

    def get_test(self):
        if self.test == None:
            X = self.get_data(self.config.test_path + "-features.pkl")
            Y = self.get_data(self.config.test_path + "-labels.pkl")
            self.test = X, Y
        else:
            X, Y = self.test
        return X, Y

    def next_batch(self, data):
        if data.lower() not in ["train", "test", "validation"]:
            raise ValueError
        func = {"train" : self.get_train, "test": self.get_test, "validation": self.get_validation}[data.lower()]
        X, Y = func()
        start = 0
        batch_size = self.config.batch_size
        tot = len(X)
        total = int(tot/ batch_size) # fix the last batch
        while start < total:
            end = start + batch_size
            x = X[start : end, :]
            y = Y[start : end, :]
            start += 1
            yield (x, y, int(total))