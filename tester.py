import learner as lrn


def get_data_filename_test():
    filename = lrn.get_data_filename()
    csv_to_dataframe_test(filename)


def csv_to_dataframe_test(filename):
    dataframe = lrn.csv_to_dataframe(filename)
    print("Raw data:")
    print(dataframe.head())
    partition_data_test(dataframe)


def partition_data_test(dataframe):
    part_ratio = 70
    train, test = lrn.partition_data(dataframe, part_ratio)
    print("Training data:")
    print(train.head())
    print("Test data:")
    print(test.head())
    categorize_test(train)

def categorize_test(train_data):
    categorized = lrn.categorize(train_data)
    print("Categorized:")
    print(categorized.head())
    clean_test(categorized)

def clean_test(categorized):
    cleaned = lrn.clean(categorized)
    print("Cleaned:")
    print(cleaned.head())
    lrn.main()


def main():
    get_data_filename_test()

main()
