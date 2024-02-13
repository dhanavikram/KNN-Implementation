import numpy as np
import pandas as pd


def train_test_val_split(df, train_size=0.70, val_size=0.15, target_col='Status'):
    train_lst = []
    val_lst = []
    test_lst = []

    groups = df.groupby(target_col)

    for label, group_df in groups:
        group_size = group_df.shape[0]

        train_split_idx = int(train_size * group_size)
        val_split_idx = int(train_split_idx + (val_size * group_size))
        train, val, test = np.split(group_df, [train_split_idx, val_split_idx])

        train_lst.append(train)
        test_lst.append(test)
        val_lst.append(val)

    final_train = pd.concat(train_lst)
    final_test = pd.concat(test_lst)
    final_val = pd.concat(val_lst)

    return final_train, final_test, final_val


class Preprocess:

    def __init__(self):
        self.df = None
        self.test_df = None
        self.ordinal_col_lst = None
        self.ordinal_codes = {}

    def convert_ordinal_to_numerical_train(self, df: pd.DataFrame, ordinal_col_lst: list) -> pd.DataFrame:

        self.df = df.copy(deep=True)
        self.ordinal_col_lst = ordinal_col_lst

        for each_col in self.ordinal_col_lst:
            temp_col = pd.Categorical(self.df[each_col])
            self.ordinal_codes[each_col] = temp_col.categories
            self.df[each_col] = temp_col.codes

        return self.df

    def convert_ordinal_to_numerical_test(self, df):

        self.test_df = df.copy(deep=True)

        for each_col in self.ordinal_col_lst:
            temp = pd.Categorical(self.test_df[each_col], categories=self.ordinal_codes[each_col])
            self.test_df[each_col] = temp.codes

        return self.test_df
