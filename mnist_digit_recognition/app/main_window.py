from PyQt6.QtWidgets import QMainWindow, QLabel, QWidget, QHBoxLayout

from mnist_digit_recognition.app.components import ModelPicker, Canvas
from mnist_digit_recognition.models.base import MlModel


class MainWindow(QMainWindow):
    title = "App"

    model_picker: ModelPicker

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.title)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20,20,20,20)
        main_layout.setSpacing(20)

        self.canvas = Canvas()
        main_layout.addWidget(self.canvas)

        model_layout = QHBoxLayout()

        model_label = QLabel("Model: ")
        model_layout.addWidget(model_label)

        self.model_picker = ModelPicker()
        model_layout.addWidget(self.model_picker)

        main_layout.addLayout(model_layout)

        # Set the layout
        w = QWidget()
        w.setLayout(main_layout)
        self.setCentralWidget(w)