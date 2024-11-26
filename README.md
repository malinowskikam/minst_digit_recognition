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
- ``app`` - run the drawing app allowing to test the models

The ``main`` script executes these steps in sequence

To run the script using poetry run:
```commandline
poetry run [script]
```

## App

The drawing app allows the user to select a model, draw a digit and make a prediction with selected model.

Run the app with ``poetry run app`` 

After running the app, select a model with the ``Model`` dropdown, draw a digit and hit ``Predict`` button to use the model
to predict the digit. You can update the drawing after the prediction or select a different model to make another
prediction.

Use the ``Clear`` button to clear the canvas and the prediction

## Available models

Currently supported models are:
- Naive Bayes (from sklearn)
- Decision Tree (from sklearn)
- Random Forest (from sklearn)
- K Nearest Neighbours (k=5, from sklearn)
- Support Vector (from sklearn)
