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

def get_data_filename():
    pass


def get_data(partition_ratio):
    data_filename = get_data_filename()
    data_array = csv_to_array(data_filename)
    return partition_data(data_array, partition_ratio)


def csv_to_array(csv_filename):
    pass


# Training #

def clean(to_clean):
    pass


def partition_data(data_array, part_ratio):
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
