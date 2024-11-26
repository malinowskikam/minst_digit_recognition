import logging

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import QMainWindow, QLabel, QWidget, QHBoxLayout, QVBoxLayout, QPushButton

from mnist_digit_recognition.app.components import ModelPicker, Canvas
from mnist_digit_recognition.app.predict_task import PredictTask

logger = logging.getLogger("mnist_digit_recognition.app")


class MainWindow(QMainWindow):
    title = "App"

    model_picker: ModelPicker
    clear_button: QPushButton
    predict_button: QPushButton
    prediction_text: QLabel

    thread_pool: QThreadPool

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.title)

        # Main window layout
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Drawing canvas
        self.canvas = Canvas()
        main_layout.addWidget(self.canvas)

        # Layout of elements on the right side of canvas
        model_layout = QVBoxLayout()
        model_layout.setSpacing(20)

        #Layout containing model label and picker
        model_select_layout = QHBoxLayout()

        model_label = QLabel("Model:")
        model_select_layout.addWidget(model_label)

        self.model_picker = ModelPicker()
        model_select_layout.addWidget(self.model_picker)

        # Layout containing clear and predict buttons
        buttons_layout = QHBoxLayout()

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self._on_clear_button_clicked)
        buttons_layout.addWidget(self.clear_button)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self._on_predict_button_clicked)
        buttons_layout.addWidget(self.predict_button)

        # Layout containing prediction label and text
        prediction_layout = QHBoxLayout()

        prediction_label = QLabel("Prediction:")
        prediction_layout.addWidget(prediction_label)

        self.prediction_text = QLabel()
        prediction_layout.addWidget(self.prediction_text)

        model_layout.addLayout(model_select_layout)
        model_layout.addLayout(buttons_layout)
        model_layout.addLayout(prediction_layout)

        main_layout.addLayout(model_layout)

        # Thread pool
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(1)

        # Set the layout
        w = QWidget()
        w.setLayout(main_layout)
        self.setCentralWidget(w)

    def _on_clear_button_clicked(self):
        logger.info("Clearing canvas and prediction")
        self.canvas.clear()
        self.prediction_text.clear()

    def _on_predict_button_clicked(self):
        logger.info("Starting prediction")
        task = PredictTask(self.model_picker.get_current_model(), self.canvas)
        task.signals.finished.connect(self._on_predict_task_finished)
        self.setDisabled(True)
        self.thread_pool.start(task)

    def _on_predict_task_finished(self, prediction: int):
        logger.info("Prediction finished, prediction: %s", prediction)
        self.prediction_text.setText(str(prediction))
        self.setDisabled(False)
