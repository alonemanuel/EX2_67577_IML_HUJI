import learner as lrn


def get_data_filename_test():
    filename = lrn.get_data_filename()
    csv_to_dataframe_test(filename)

def csv_to_dataframe_test(filename):
    dataframe = lrn.csv_to_dataframe(filename)
    partition_data_test(dataframe)

def partition_data_test(dataframe):
    part_ratio = 70
    train, test = lrn.partition_data(dataframe, part_ratio)


def main():
    get_data_filename_test()
