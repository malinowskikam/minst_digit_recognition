import logging

from PyQt6.QtCore import pyqtSignal, QObject, QRunnable

from mnist_digit_recognition.app.components import Canvas
from mnist_digit_recognition.data import InputData
from mnist_digit_recognition.models.base import MlModel

logger = logging.getLogger("mnist_digit_recognition.app")

class PredictTaskSignals(QObject):
    finished = pyqtSignal(int)
    error = pyqtSignal(str)


class PredictTask(QRunnable):
    model: MlModel
    canvas: Canvas
    signals: PredictTaskSignals

    def __init__(self, model: MlModel, canvas):
        super().__init__()
        self.model = model
        self.canvas = canvas
        self.signals = PredictTaskSignals()

    def run(self):
        try:
            if not self.model.ready():
                logger.info("Loading model")
                self.model.load()

            logger.info("Loading image")
            image = self.canvas.get_image().reshape(1, -1)
            input_data = InputData()
            input_data.test_x = image

            logger.info("Predicting using %s", self.model.name)
            prediction = self.model.predict(input_data)[0]
            logger.info("Prediction: %s", prediction)

            self.signals.finished.emit(prediction)
        except Exception as e:
            logger.error("Error during predict task", exc_info=e)
            error = str(e)
            self.signals.error.emit(error)
