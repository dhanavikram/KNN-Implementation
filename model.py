import numpy as np
import pandas as pd
import statistics
from itertools import zip_longest


def euclidean_distance(point_a: np.array, point_b: np.array):
    """
    Function to compute Euclidean distance between two numpy arrays.
    :param point_a: Array one representing an observation in a dataset
    :param point_b: Array two representing an observation in a dataset
    :return: Distance between the given arrays
    """
    return np.sqrt(np.sum((point_a - point_b) ** 2, axis=0))


def hamming_distance(str_array_a, str_array_b):
    """
    Function to compute hamming distance between two arrays of strings
    :param str_array_a: Array one representing the categorical values of an observation in a dataset
    :param str_array_b: Array two representing the categorical values of an observation in a dataset
    :return: Distance between the given arrays
    """

    def find_hamming_dist(string_a, string_b):
        return sum(a != b for a, b in zip_longest(string_a, string_b))

    return np.sum([find_hamming_dist(string_a, string_b) for string_a, string_b in zip(str_array_a, str_array_b)])


class KNN:
    """
    Class to predict label based on KNN algorithm.
    """
    def __init__(self, k):
        """
        Constructor method. Instantiates class-scope variables.
        :param k: Number of neighbors to consider
        """
        self.k = k

        # Placeholders
        self.x = None
        self.y = None
        self.total_length = None
        self.numerical_cols = None
        self.ordinal_cols = None
        self.ordinal_codes = {}
        self.categorical_cols = None
        self.numerical_x_train = None
        self.categorical_x_train = None
        self.numerical_x_test = None
        self.categorical_x_test = None

    def fit(self, x, y, numerical_cols, ordinal_cols, categorical_cols):
        """
        Fits the instance on the training dataset. Segregates the arrays into numerical and
        categorical_columns
        :param x: Dataframe containing all the independent variables
        :param y: Series containing the class labels
        :param numerical_cols: List of numerical columns
        :param ordinal_cols: List of ordinal columns
        :param categorical_cols: List of categorical columns
        :return: None
        """

        if x.shape[0] != y.shape[0]:
            raise ValueError("Input features and Labels are of not same length")

        self.x = x
        self.y = y
        self.total_length = self.x.shape[0]
        self.numerical_cols = numerical_cols
        self.ordinal_cols = ordinal_cols
        self.categorical_cols = categorical_cols

        for each_col in self.ordinal_cols:
            temp_col = pd.Categorical(self.x[each_col])
            self.ordinal_codes[each_col] = temp_col.categories
            self.x[each_col] = temp_col.codes

        self.numerical_x_train = x[numerical_cols + ordinal_cols].to_numpy()
        self.categorical_x_train = x[categorical_cols].to_numpy()

    def predict(self, x_test):
        """
        Predicts the labels for given test samples
        :param x_test: Dataframe of independent variables of test data
        :return: Array of predicted labels
        """
        for each_col in self.ordinal_cols:
            temp = pd.Categorical(x_test[each_col], categories=self.ordinal_codes[each_col])
            x_test[each_col] = temp.codes

        self.numerical_x_test = x_test[self.numerical_cols + self.ordinal_cols].to_numpy()
        self.categorical_x_test = x_test[self.categorical_cols].to_numpy()

        predicted_y = [self._predict_for_each_instance(
            self.numerical_x_test[i], self.categorical_x_test[i]) for i in range(self.total_length)]

        return np.array(predicted_y)

    def _predict_for_each_instance(self, x_test_numerical, x_test_categorical):
        """
        Predicts class label for each instance
        :param x_test_numerical: Single sample containing values from numerical and ordinal columns
        :param x_test_categorical: Single sample containing values from categorical columns
        :return: Predicted class label for the provided sample/instance
        """
        euclidean_dist = np.array([euclidean_distance(x_test_numerical, each_train_sample)
                                   for each_train_sample in self.numerical_x_train])
        categorical_dist = np.array([hamming_distance(x_test_categorical, each_train_sample)
                                     for each_train_sample in self.categorical_x_train])

        total_dist = np.sum([euclidean_dist, categorical_dist], axis=0)
        top_k_idx = np.argsort(total_dist)[0:self.k]
        labels = np.take(self.y, top_k_idx)

        return statistics.mode(labels)
