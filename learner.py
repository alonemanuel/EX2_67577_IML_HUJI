"""
Author: Alon Emanuel
For:    Ex2 in 67577: IML in HUJI
Brief:  First learning project.
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib as plt


# Helpers #


def get_data(partition_ratio):
    """

    :param partition_ratio: ratio for partitioning between training set and test set.
    :return: a tuple of training data and test data (both as np arrays)
    """
    data_filename = get_data_filename()
    dataframe = csv_to_dataframe(data_filename)
    return partition_data(dataframe, partition_ratio)


def get_data_filename():
    """
    :return: the name of the file in which the data is stored.
    """
    filename = './Data/kc_house_data.csv'
    return filename


def csv_to_dataframe(csv_filename):
    """
    :param csv_filename: csv type data.
    :return: pandas DataFrame
    """
    return pd.read_csv(csv_filename)


def partition_data(dataframe, part_ratio):
    """
    :param dataframe: a pandas DataFrame holding the data
    :param part_ratio: partition ratio
    :return: training set and test set, both as DataFrame
    """
    n_of_samples = dataframe.shape[0]
    partition_idx = n_of_samples * part_ratio // 100
    return dataframe.iloc[:, :partition_idx], dataframe.iloc[:, partition_idx:]


# Training #

def clean(to_clean):
    pass


def categorize(to_categorize):
    pass


def preprocess(csv_filename, train_data):
    categorize(train_data)
    clean(train_data)


def train(train_data):
    preprocess(train_data)


# Final Output #

def plotter(train_error, test_error):
    pass


def get_train_error(learner, train_data):
    pass


def get_test_error(learner, test_data):
    pass


def main():
    for partition_ratio in range(100):
        train_data, test_data = get_data(partition_ratio)
        learner = train(train_data)
        train_error = get_train_error(learner, train_data)
        test_error = get_test_error(learner, test_data)
        plotter(train_error, test_error)
