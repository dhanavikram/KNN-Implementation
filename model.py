import numpy as np
import statistics


def euclidean_distance(point_a, point_b):
    return np.sqrt(np.sum((point_a - point_b) ** 2, axis=0))


def hamming_distance(string_a, string_b):
    pass


class KNN:

    def __init__(self, k):
        self.k = k

        # Placeholders
        self.x = None
        self.y = None
        self.total_length = None
        self.numerical_cols = None
        self.ordinal_cols = None
        self.categorical_cols = None
        self.numerical_x_train = None
        self.categorical_x_train = None
        self.numerical_x_test = None
        self.categorical_x_test = None

    def fit(self, x, y, numerical_cols, ordinal_cols, categorical_cols):

        if x.shape[0] != y.shape[0]:
            raise ValueError("Input features and Labels are of not same length")

        self.x = x
        self.y = y
        self.total_length = self.x.shape[0]
        self.numerical_cols = numerical_cols
        self.ordinal_cols = ordinal_cols
        self.categorical_cols = categorical_cols

        self.numerical_x_train = x[numerical_cols + ordinal_cols]
        self.categorical_x_train = x[categorical_cols]

    def predict(self, x_test):
        self.numerical_x_test = x_test[self.numerical_cols + self.ordinal_cols].to_numpy()
        self.categorical_x_test = x_test[self.categorical_cols].to_numpy()

        predicted_y = [self._predict_for_each_instance(
            self.numerical_x_test[i], self.categorical_x_test[i]) for i in range(self.total_length)]

        return np.array(predicted_y)

    def _predict_for_each_instance(self, x_test_numerical, x_test_categorical):
        euclidean_dist = np.array([euclidean_distance(x_test_numerical, each_train_sample)
                                   for each_train_sample in self.numerical_x_train])
        categorical_dist = np.array([hamming_distance(x_test_categorical, each_train_sample)
                                     for each_train_sample in self.categorical_x_train])

        total_dist = np.sum(euclidean_dist, categorical_dist)
        top_k_idx = np.argsort(total_dist)[0:self.k]
        labels = np.take(self.y, top_k_idx)

        return statistics.mode(labels)
