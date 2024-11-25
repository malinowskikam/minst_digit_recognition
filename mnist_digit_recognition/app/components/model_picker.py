from PyQt6.QtWidgets import QComboBox

from mnist_digit_recognition.models import get_models
from mnist_digit_recognition.models.base import MlModel


class ModelPicker(QComboBox):
    default_option = "cnn"
    models: dict[str, MlModel]

    def __init__(self):
        super().__init__()

        self.models = get_models()

        for key, model in self.models.items():
            self.addItem(key, userData=model)

        if self.default_option in self.models:
            self.setCurrentText(self.default_option)

    def get_current_model(self):
        return self.models[self.currentText()]
