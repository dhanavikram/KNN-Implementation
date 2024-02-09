class EvaluateBinaryClass:
    """
    Class to evaluate the results of a Binary Classification algorithm.
    """

    def __init__(self, y_actual, y_predicted):
        """
        Constructor method. Calculates the values of Confusion matrix based on the arrays passed
        :param y_actual: Actual class labels
        :param y_predicted: Predicted class labels
        """
        self.y_actual = y_actual
        self.y_predicted = y_predicted

        self.length_of_y = len(self.y_actual)

        if self.length_of_y != len(self.y_predicted):
            raise ValueError("Arrays to be compared are of different lengths")

        self.true_positive = 0
        self.false_positive = 0
        self.true_negative = 0
        self.false_negative = 0

        for i in range(self.length_of_y):
            if self.y_actual[i] == 1:
                if self.y_predicted[i] == 1:
                    self.true_positive += 1
                elif self.y_predicted[i] == 0:
                    self.false_negative += 1
            else:
                if self.y_predicted[i] == 1:
                    self.false_positive += 1
                elif self.y_predicted[i] == 0:
                    self.true_negative += 1

        self.precision = self.true_positive / (self.true_positive + self.false_positive)
        self.sensitivity = self.true_positive / (self.true_positive + self.false_negative)
        self.recall = self.sensitivity
        self.specificity = self.true_negative / (self.true_negative + self.false_positive)

    def get_accuracy(self):
        """
        Method to get the accuracy for the provided labels
        :return: Accuracy value
        """
        no_of_correct_preds = self.true_positive + self.true_negative

        return no_of_correct_preds / self.length_of_y

    def get_balanced_accuracy(self):
        """
        Method to get the balanced accuracy for the provided labels
        :return: Balanced Accuracy value
        """
        balanced_accuracy = (self.sensitivity + self.specificity) / 2

        return balanced_accuracy

    def get_f1_score(self):
        """
        Method to get the F1 score for the provided labels
        :return: F1 Score
        """
        f1_score = (2 * self.precision * self.recall) / (self.recall + self.precision)

        return f1_score
