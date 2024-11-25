from PyQt6.QtWidgets import QComboBox

from mnist_digit_recognition.models import get_models
from mnist_digit_recognition.models.base import MlModel


class ModelPicker(QComboBox):
    _default_option = "cnn"
    _models: dict[str, MlModel]

    def __init__(self):
        super().__init__()

        self._models = get_models()

        for key, model in self._models.items():
            self.addItem(key, userData=model)

        if self._default_option in self._models:
            self.setCurrentText(self._default_option)

    def get_current_model(self):
        return self._models[self.currentText()]
