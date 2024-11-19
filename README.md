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

- ``fit`` - train all models and save them in the ``./models`` directory
- ``score`` - calculate accuracy of the models
- ``confusion_matrix`` - calculate confusion matrices for the models

The ``main`` script executes these steps in sequence

To run the script using poetry run:
```commandline
poetry run [script]
```

## App

## Available models

Currently supported models are:
- Naive Bayes (from sklearn)
- Decision Tree (from sklearn)
- Random Forest (from sklearn)
- K Nearest Neighbours (k=5, from sklearn)
- Support Vector (from sklearn)
