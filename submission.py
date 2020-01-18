from keras.engine.saving import load_model

from data import get_test_data
from nn import predict_nn
import nn
from util import render_submission

data = get_test_data()
nn.model = load_model("models\\convolutional_neural_network_10_86.h5")
prediction = predict_nn(data)
render_submission(prediction)
