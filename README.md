# MNIST_digit_recognition

A small project comparing various machine learning techniques for recognizing handwritten digits, using the MNIST dataset.

## Installation

Install this project using poetry
```commandline
petry install
```

## Training data

This project is based on [this](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) exercise from Kaggle.com

To run this project, you need to download the data (``train.csv``) from this exercise and put it under ``./data`` directory

## Commands

This package offers several executable scripts:

- ``mnist_digit_recognition.fit`` - train all models and save them in the ``./models`` directory
- ``mnist_digit_recognition.score`` - calculate accuracy of the models
- ``mnist_digit_recognition.confusion_matrix`` - calculate confusion matrices for the models

The ``mnist_digit_recognition.__main__`` package executes these steps in sequence

## App

## Available models
