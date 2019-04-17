"""
Author: Alon Emanuel
For:    Ex2 in 67577: IML in HUJI
Brief:  First learning project.
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib as plt

# Static Vars
missing_values = ["n/a", "na", "--", "nan"]


# Math
def get_sigsigt_mat(xmat):
    """
    :param xmat: matrix to manipulate on
    :return: the \Sigma\Sigma.T matrix, where Sigma is the Sigma from X's svd composition.
    """
    u, d, vh = np.linalg.svd(xmat)
    return np.matmul(d, d.T)

def get_sqloss(wvec, xvec, y):
    """
    :param wvec: weight vector      (d x 1)
    :param xvec: feature vector     (d x 1)
    :param y: true label            (scalar)
    :return: square loss
    """
    inprod = np.inner(wvec, xvec)
    delta = np.subtract(inprod, y)
    deltasq = np.square(delta)
    return deltasq


def get_empirical_mse(wvec, xvecs, yvec):
    """
    :param wvec: weights vector             (d x 1)
    :param xvecs: feature vectors as cols   (d x m)
    :param yvec: true label vector          (m x 1)
    :return: empirical mean square error    (scalar)
    """
    innervec = np.matmul(xvecs.T, wvec)
    deltavec = np.subtract(innervec, yvec)
    deltasq = np.square(deltavec)
    mse = np.mean(deltasq)
    return mse


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
    return pd.read_csv(csv_filename, na_values=missing_values)


def partition_data(df, part_ratio):
    """
    :param dataframe: a pandas DataFrame holding the data
    :param part_ratio: partition ratio
    :return: training set and test set, both as DataFrame
    """
    n_of_samples = df.shape[0]
    partition_idx = n_of_samples * part_ratio // 100
    df = df.sample(frac=1).reset_index(drop=True)  # shuffle df
    return df.iloc[:partition_idx], df.iloc[partition_idx:]


# Training #
def categorize(to_categorize):
    return pd.get_dummies(to_categorize, prefix='zip', prefix_sep='_', columns=['zipcode'])


def clean(to_clean):
    return to_clean


def preprocess(train_data):
    categorized = categorize(train_data)
    cleaned = clean(categorized)
    return cleaned


def train(train_data):
    preprop = preprocess(train_data)
    features_to_drop = ['id', 'date', 'price']
    X = preprop.drop(features_to_drop, axis=1)
    Y = preprop['price']
    return X, Y


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
