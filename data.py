import pandas
from keras.utils import to_categorical
from pandas import to_numeric
from sklearn.model_selection import train_test_split


test_data_file = "data\\test.csv"
train_data_file = "data\\train.csv"


def load_train_data():
    dataframe = pandas.read_csv(train_data_file).apply(to_numeric)

    x = dataframe.drop("label", axis=1).values
    y = dataframe["label"].values

    return train_test_split(x, y, test_size=0.1, random_state=1234)


def get_training_data_4d():

    data = list(load_train_data())

    data[0] = (data[0] / 255.0).reshape(-1,28,28,1)
    data[1] = (data[1] / 255.0).reshape(-1, 28, 28, 1)

    data[2] = to_categorical(data[2],num_classes=10)
    data[3] = to_categorical(data[3], num_classes=10)

    return tuple(data)


def get_training_data_2d():
    return load_train_data()


def get_submission_data():
    dataframe = pandas.read_csv(test_data_file).apply(to_numeric)
    return (dataframe.values / 255.0).reshape(-1,28,28,1)