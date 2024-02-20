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
        self.target_col = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.numerical_x_train = None
        self.categorical_x_train = None
        self.x_test = None
        self.y_actual = None
        self.y_predicted = None
        self.numerical_x_test = None
        self.categorical_x_test = None

    def fit(self, df: pd.DataFrame, target_col: str, numerical_cols: list, categorical_cols: list):
        """
        Fits the instance on the training dataset. Segregates the arrays into numerical and
        categorical_columns
        :param df: Dataframe containing KNN training samples.
        :param target_col: Name of the column containing the target samples.
        :param numerical_cols: List of numerical columns
        :param categorical_cols: List of categorical columns
        :return: None
        """

        self.x = df.drop([target_col], axis=1)
        self.y = df[target_col]
        self.total_length = self.x.shape[0]
        self.target_col = target_col
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols

        self.numerical_x_train = self.x[self.numerical_cols].to_numpy()
        self.categorical_x_train = self.x[categorical_cols].to_numpy()

    def predict(self, test_df: pd.DataFrame) -> np.array:
        """
        Predicts the labels for given test samples
        :param test_df: Dataframe of test data
        :return: Array of predicted labels
        """

        self.x_test = test_df.drop([self.target_col], axis=1)
        self.y_actual = test_df[self.target_col].to_numpy()
        self.numerical_x_test = self.x_test[self.numerical_cols].to_numpy()
        self.categorical_x_test = self.x_test[self.categorical_cols].to_numpy()

        predicted_y = [self._predict_for_each_instance(
            self.numerical_x_test[i], self.categorical_x_test[i]) for i in range(test_df.shape[0])]

        self.y_predicted = np.array(predicted_y)

        return self.y_predicted

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

    def get_accuracy(self):
        """
        Method to get the accuracy of the model's prediction
        :return: Accuracy value ranging from 0 to 1
        """
        numerator = np.sum(self.y_predicted == self.y_actual)
        denominator = len(self.y_predicted)
        accuracy = numerator/denominator
        return accuracy

    def get_balanced_accuracy(self):
        """
        Method to get the Balanced accuracy metric. Used when class imbalance exists.
        :return: Balanced Accuracy value ranging from 0 to 1
        """
        unique_classes, class_counts = np.unique(self.y_actual, return_counts=True)
        class_accuracies = []

        for each_class in unique_classes:
            true_positive = np.sum((self.y_predicted == each_class) & (self.y_actual == each_class))
            total_samples_class = class_counts[each_class]

            class_accuracy = true_positive / total_samples_class if total_samples_class != 0 else 0
            class_accuracies.append(class_accuracy)

        balanced_accuracy = np.mean(class_accuracies)
        return balanced_accuracy

    def get_f1_score(self, return_precision_and_recall: bool = False):
        """
        Method to get F1 score of the model's prediction. Computes precision and recall and returns their harmonic mean.
        :param return_precision_and_recall: If set true, returns the precision and recall values along with F1 Score.
        :return: F1 score value ranging from 0 to 1
        """
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0

        for actual, predicted in zip(self.y_actual, self.y_predicted):
            true_positive += actual and predicted
            false_positive += not actual and predicted
            true_negative += not actual and not predicted
            false_negative += actual and not predicted

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        if return_precision_and_recall:
            return f1_score, precision, recall
        else:
            return f1_score
