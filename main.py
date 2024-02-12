import pandas as pd

from model import KNN
from utils import train_test_val_split, Preprocess
from evaluate import EvaluateBinaryClass

data = pd.read_csv('Breast_Cancer.csv')

# Split data to train, validation and test datasets
train, val, test = train_test_val_split(data, train_size=0.7, val_size=0.15)

# Preprocess the train dataset - convert ordinal columns to numerical columns
ordinal_cols = ['T Stage ', 'N Stage', '6th Stage', 'differentiate', 'Grade',
                'A Stage', 'Estrogen Status', 'Progesterone Status', 'Status']
pre_processor = Preprocess()
pre_proc_train = pre_processor.convert_ordinal_to_numerical_train(train, ordinal_col_lst=ordinal_cols)

# Apply the KNN model on pre-processed dataset.
target_col = 'Status'
categorical_cols = ['Race', 'Marital Status']
num_cols = [column for column in pre_proc_train.columns if column not in categorical_cols+[target_col]]

# Fit model on training dataset
model = KNN(k=3)
model.fit(df=pre_proc_train, target_col=target_col, numerical_cols=num_cols , categorical_cols=categorical_cols)

# Predict for val dataset
pre_proc_val = pre_processor.convert_ordinal_to_numerical_test(val)
result = model.predict(pre_proc_val)

model.get_accuracy()
