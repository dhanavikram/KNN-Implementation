import numpy as np
import pandas as pd


def shuffle_dataframe(df):
    """
    Function to shuffle the given dataframe.
    :param df: dataframe which is to be shuffled.
    :return: Shuffled dataframe
    """
    indices = np.arange(df.shape[0])
    shuffled_indices = np.random.RandomState(123).permutation(indices)
    return df.iloc[shuffled_indices, :].reset_index(drop=True)


def train_test_val_split(df, train_size: float = 0.70, val_size: float = 0.15,
                         target_col: str = 'Status', concat_train_val: bool = False):
    """
    Function used to split the given dataset into training, validation and test datasets in a stratified manner.
    :param df: Dataset to be split
    :param train_size: Size of the training dataset
    :param val_size: Size of the validation dataset
    :param target_col: Column used to stratify the split
    :param concat_train_val: If set True,
    will concatenate train and validation dataframes and return it along with test dataframe
    :return: Train, Validation and Test Dataframes
    """
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
        val_lst.append(val)
        test_lst.append(test)

    final_train = shuffle_dataframe(pd.concat(train_lst, ignore_index=True))
    final_val = shuffle_dataframe(pd.concat(val_lst, ignore_index=True))
    final_test = shuffle_dataframe(pd.concat(test_lst, ignore_index=True))

    if concat_train_val:
        return pd.concat([final_train, final_val], ignore_index=True), final_test

    return final_train, final_val, final_test


class Preprocessor:
    """
    Class used to preprocess the given training and validation/test dataframes
    """

    def __init__(self):
        """
        Constructor method. Instantiates the class and required attributes.
        """
        self.df = None
        self.test_df = None
        self.ordinal_col_lst = None
        self.ordinal_codes = {}

    def convert_ordinal_to_numerical_train(self, df: pd.DataFrame, ordinal_col_lst: list) -> pd.DataFrame:
        """
        Method used to convert the ordinal categorical columns to numerical columns in training dataframe.
        :param df: Train dataframe
        :param ordinal_col_lst: List of ordinal columns
        :return: Pre-processed Dataframe
        """

        self.df = df.copy(deep=True)
        self.ordinal_col_lst = ordinal_col_lst

        for each_col in self.ordinal_col_lst:
            temp_col = pd.Categorical(self.df[each_col])
            self.ordinal_codes[each_col] = temp_col.categories
            self.df[each_col] = temp_col.codes

        return self.df

    def convert_ordinal_to_numerical_test(self, df) -> pd.DataFrame:
        """
        Method used to convert the ordinal categorical columns to numerical columns in test or validation dataframe.
        :param df: Test/Validation Dataframe
        :return: Pre-processed Dataframe
        """
        self.test_df = df.copy(deep=True)

        for each_col in self.ordinal_col_lst:
            temp = pd.Categorical(self.test_df[each_col], categories=self.ordinal_codes[each_col])
            self.test_df[each_col] = temp.codes

        return self.test_df
