"""
data loading package
"""

import pathlib

import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from minst_digit_recognition.data.joblibSerializationMixin import JoblibModelSerializationMixin


class TrainingData:
    """
    Data loader for ML models
    """
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray

    def __init__(self, data_dir="data", data_file="train.csv", test_size=0.1):
        self._data_dir = data_dir
        self._data_file = data_file
        self._test_size = test_size

        file_ext = data_file.rsplit(".", 1)[-1]
        match file_ext:
            case "csv":
                (self.train_x, self.test_x, self.train_y, self.test_y) = self._load_data_csv()
            case _:
                raise AttributeError(f"Unknown file extension: {file_ext}")

    def _load_data_csv(self) -> list:
        train_data_path = pathlib.Path(self._data_dir, self._data_file)
        dataframe = pd.read_csv(train_data_path).apply(pd.to_numeric)

        x = dataframe.drop("label", axis=1).values
        y = dataframe["label"].values

        return train_test_split(x, y, test_size=self._test_size, random_state=1234)


# def get_training_data_4d():
#
#     data = list(load_train_data())
#
#     data[0] = (data[0] / 255.0).reshape(-1,28,28,1)
#     data[1] = (data[1] / 255.0).reshape(-1, 28, 28, 1)
#
#     data[2] = to_categorical(data[2],num_classes=10)
#     data[3] = to_categorical(data[3], num_classes=10)
#
#     return tuple(data)
#
#
# def get_training_data_2d():
#     return load_train_data()
#
#
# def get_submission_data():
#     dataframe = pandas.read_csv(test_data_file).apply(to_numeric)
#     return (dataframe.values / 255.0).reshape(-1,28,28,1)
