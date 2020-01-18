import pandas
from keras.utils import to_categorical
from pandas import to_numeric
from sklearn.model_selection import train_test_split


test_data_file = "data\\test.csv"
train_data_file = "data\\train.csv"


def get_training_data():
    dataframe = pandas.read_csv(train_data_file).apply(to_numeric)

    x = (dataframe.drop("label", axis=1).values / 255.0).reshape(-1,28,28,1)
    y = to_categorical(dataframe["label"].values, num_classes=10)

    return train_test_split(x, y, test_size=0.1, random_state=1234)

def get_test_data():
    dataframe = pandas.read_csv(test_data_file).apply(to_numeric)
    return (dataframe.values / 255.0).reshape(-1,28,28,1)