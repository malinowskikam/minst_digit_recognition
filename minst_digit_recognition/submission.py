from keras.engine.saving import load_model

from data import get_submission_data
from util import render_submission

data = get_submission_data()
model = load_model("models/convolutional_neural_network_30_86.h5")
prediction = model.predict(data)
render_submission(prediction)
