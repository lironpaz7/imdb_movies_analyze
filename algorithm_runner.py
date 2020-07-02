from sklearn import neighbors
from computes import *


class AlgorithmRunner:

    def __init__(self, data, algorithm, k=10):
        """
        Runs the specified algorithm on processed data and calculates scores.
        :param algorithm: String represent algorithm to use: 'KNN' or 'Rocchio'
        :param k: Optional - initializes 'KNN' algorithm number of neighbors (default = 10).
        """
        self._name = algorithm
        self._data = data
        if algorithm == "KNN":
            self.algorithm = neighbors.KNeighborsClassifier(n_neighbors=k)
        elif algorithm == "Rocchio":
            self.algorithm = neighbors.NearestCentroid()
        else:
            print("Please enter one of : KNN or Rocchio")
        self._precision = 0
        self._recall = 0
        self._accuracy = 0

    def fit(self, x_train, y_train):
        """
        Wrapper method, fits the data by using the built in algorithm fit method.
        :param x_train: Data for training.
        :param y_train: Labels for training.
        """
        self.algorithm.fit(x_train, y_train)

    def predict(self, x_test):
        """
        Wrapper method, predicts data labels by using built in algorithm predict method.
        :param x_test: Data for testing.
        :return: Predicted vector.
        """
        return self.algorithm.predict(x_test)

    def run(self, print_data=True):
        """
        Runs the classifier and computes the precision, recall and accuracy.
        :param data: Object of Data class.
        :param algorithm: Object of AlgorithmRunner class.
        :param print_data: Boolean flag that prints the data.
        """
        kfolds = self._data.split_to_k_folds()
        folds = 0
        for train_index, test_index in kfolds:
            x_train, x_test = self._data.p_data[train_index], self._data.p_data[test_index]
            y_train, real = self._data.scores[train_index], self._data.scores[test_index]
            self.fit(x_train, y_train)
            predicted = self.predict(x_test)
            self._precision += compute_precision(real, predicted)
            self._recall += compute_recall(real, predicted)
            self._accuracy += compute_accuracy(real, predicted)
            folds += 1
        self._precision /= folds
        self._recall /= folds
        self._accuracy /= folds
        if print_data:
            print(f"{self._name} classifier: {self._precision}, {self._recall}, {self._accuracy}")
