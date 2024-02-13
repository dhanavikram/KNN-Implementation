import numpy as np
import pandas as pd

from model import KNN
from utils import train_test_val_split, Preprocess

data = pd.read_csv('Breast_Cancer.csv')

# Split data to train, validation and test datasets
train, val, test = train_test_val_split(data, train_size=0.7, val_size=0.15)

# Preprocess the train dataset - convert ordinal columns to numerical columns
ordinal_cols = ['T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade',
                'A Stage', 'Estrogen Status', 'Progesterone Status', 'Status']
pre_processor = Preprocess()
pre_proc_train = pre_processor.convert_ordinal_to_numerical_train(train, ordinal_col_lst=ordinal_cols)

# Preprocess the validation dataset
pre_proc_val = pre_processor.convert_ordinal_to_numerical_test(val)

# Apply the KNN model on pre-processed dataset.
target_col = 'Status'
categorical_cols = ['Race', 'Marital Status']
num_cols = [column for column in pre_proc_train.columns if column not in categorical_cols+[target_col]]

# Fit model on training dataset for different k values

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
accuracies = []
balanced_accuracies = []
f1_scores = []


for k in k_values:
    model = KNN(k=k)
    model.fit(df=pre_proc_train, target_col=target_col, numerical_cols=num_cols, categorical_cols=categorical_cols)

    # Predict for val dataset
    result = model.predict(pre_proc_val)

    # Print Accuracy
    accuracies.append(model.get_accuracy())
    balanced_accuracies.append(model.get_balanced_accuracy())
    f1_scores.append(model.get_f1_score())

print(f"The K value {k_values[np.argmax(accuracies)]} provides the best accuracy")
print(f"The K value {k_values[np.argmax(balanced_accuracies)]} provides the best balanced accuracy")
print(f"The K value {k_values[np.argmax(f1_scores)]} provides the best F1 Score")
