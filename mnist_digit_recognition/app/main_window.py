from PyQt6.QtWidgets import QMainWindow, QLabel, QWidget, QHBoxLayout

from mnist_digit_recognition.app.components import ModelPicker
from mnist_digit_recognition.models.base import MlModel


class MainWindow(QMainWindow):
    title = "App"

    model_picker: ModelPicker

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.title)

        self.model_picker = ModelPicker()

        model_label = QLabel("Model: ")

        model_layout = QHBoxLayout()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_picker)

        # Set the layout
        w = QWidget()
        w.setLayout(model_layout)
        self.setCentralWidget(w)