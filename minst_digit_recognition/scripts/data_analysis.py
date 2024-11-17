"""
script for calculating various metrics of the training data
"""


def main():
    import pandas
    import numpy as np
    import cv2

    train_data_file = "data/train.csv"
    dataframe = pandas.read_csv(train_data_file)
    dataframe_num = dataframe.apply(pandas.to_numeric, errors='coerce')

    print(f"Shape: {dataframe.shape}")
    print(f"null values: {dataframe.isnull().sum().max() > 0}")
    print(f"non numeric: {dataframe_num.isnull().sum().max() > 0}")
    print(f"outside range: {dataframe_num.where(np.logical_and(dataframe_num >= 0, dataframe_num <= 255)).isnull().sum().max() > 0}")
    print(f"labels:\n{dataframe['label'].value_counts().sort_index()}")
    mean = dataframe.drop('label', axis=1).mean(axis=0).apply(np.round).apply(int).values.reshape(28, 28)
    mean = mean * -1 + 255
    cv2.imwrite("data/mean.png", cv2.resize(mean, (560, 560), interpolation=cv2.INTER_NEAREST))


if __name__ == "__main__":
    main()
