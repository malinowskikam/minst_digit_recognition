from keras.engine.saving import load_model

from data import get_test_data
from util import render_submission

data = get_test_data()
model = load_model("models\\convolutional_neural_network_10_86.h5")
prediction = model.predict(data)
render_submission(prediction)
