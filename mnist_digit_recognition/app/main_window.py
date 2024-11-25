from PyQt6.QtWidgets import QMainWindow

from mnist_digit_recognition.app.components import ModelPicker
from mnist_digit_recognition.models.base import MlModel


class MainWindow(QMainWindow):
    title = "App"

    model_picker: ModelPicker

    def __init__(self):
        super().__init__()

        self.model_picker = ModelPicker()

        self.setWindowTitle(self.title)
        self.setCentralWidget(self.model_picker)
