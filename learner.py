"""
Author: Alon Emanuel
For:    Ex2 in 67577: IML in HUJI
Brief:  First learning project.
"""

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import logging, sys

# Static Vars
missing_values = ["n/a", "na", "--", "nan"]


# Math
def get_sigsigt_mat(xmat: np.ndarray):
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


def get_empirical_loss(f_hat, training_data):
    """
    :param f_hat: estimation function       (d x 1)
    :param xvecs: feature vectors as *rows* (m x d)
    :param yvec: true label vector          (m x 1)
    :return: empirical mean square error    (scalar)
    """
    xvecs, yvec = training_data
    # vecd_f = np.vectorize(f_hat)
    # print(xvecs.shape)
    # print(yvec.shape)
    f_on_data = f_hat(xvecs)
    # print(f_on_data.shape, yvec.shape)
    deltavec = np.subtract(f_on_data, yvec)
    deltasq = np.square(deltavec)
    # print(deltasq)
    return np.mean(deltasq)


def get_erm_wvec(xvecs, yvec):
    """
    :param xvecs: feature vectors as *rows*      (m X d)
    :param yvec: true labels vector              (d X 1)
    :return:
    """
    # print(xvecs)
    xpinv = np.linalg.pinv(xvecs)
    return np.matmul(xpinv, yvec)


# Helpers #

def get_preprop_data(part_ratio: int) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                               Tuple[np.ndarray, np.ndarray]]:
    data = get_data()
    preprocessed = preprocess(data)
    # preprocessed.to_csv(path_or_buf='./datatest.csv') # TODO: debug
    train_data, test_data = partition_data(preprocessed, part_ratio)
    train_x = train_data.drop("price", axis=1)
    train_y = train_data['price']
    test_x = test_data.drop("price", axis=1)
    test_y = test_data['price']
    return (train_x, train_y), (test_x, test_y)


def get_data():
    """
    :param partition_ratio: ratio for partitioning between training set and test set.
    :return: a tuple of training data and test data (both as np arrays)
    """
    data_filename = get_data_filename()
    dataframe = csv_to_dataframe(data_filename)
    return dataframe


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
    return pd.get_dummies(to_categorize, prefix='zip', prefix_sep='_',
                          columns=['zipcode'])


def clean(to_clean):
    features_to_drop = ['id', 'date']
    dropped = to_clean.drop(features_to_drop, axis='columns')
    numeric = dropped.apply(pd.to_numeric, errors='coerce')
    return numeric.dropna()


def preprocess(data):
    data = categorize(data)
    return clean(data)


def train(train_data):
    """
    :param train_data: m samples.
    :return: learner function that maps a d-feature vector to a value in R.
    """
    X, Y = train_data
    weights = get_erm_wvec(X, Y)
    print('weights', weights.shape)
    print(weights)
    # print('weights', weights.shape)
    # print('X', X.shape)
    learner = lambda x: np.matmul(x, weights)
    return learner  # TODO: does the learner accept feature vectors before preprocessing?


# Final Output #

def plotter(train_error, test_error):
    x_axis = np.arange(1, 100)
    plt.plot(x_axis, train_error, label="Training Error")
    plt.plot(x_axis, test_error, label="Test Error")
    plt.xlabel("Partition Ratio")
    plt.ylabel("Error rate")
    plt.yscale('log')
    plt.legend()
    plt.savefig('./plswork.png')


def get_train_error(learner, train_data):
    return get_empirical_loss(learner, train_data)


def get_test_error(learner: np.ndarray, test_data: Tuple[Tuple, Tuple]) -> \
        float:
    # TODO: a copy of the above function... not good
    return get_empirical_loss(learner, test_data)


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.debug("Partitioning")
    partition_ratios = np.arange(1, 100)
    logging.debug("Getting preprocessed data")
    train_data, test_data = np.vectorize(get_preprop_data)(partition_ratios)
    # train_data, test_data=np.vectorize(get_preprop_data)(np.arange(1, 101))
    logging.debug("Training data")
    learner = np.vectorize(train)(train_data)
    # print(learner.shape) #TODO: DEBUG
    logging.debug("Getting errors")
    train_error = np.vectorize(get_train_error)(learner, train_data)
    test_error = np.vectorize(get_test_error)(learner, test_data)
    logging.debug("Plotting")
    plotter(test_error, train_error)


main()
