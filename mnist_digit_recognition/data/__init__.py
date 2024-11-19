"""
data loading package
"""

from mnist_digit_recognition.data.trainingData import TrainingData

# import pathlib
#
# import numpy as np
# import pandas as pd
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split


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
