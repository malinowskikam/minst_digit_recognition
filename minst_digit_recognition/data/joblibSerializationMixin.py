from abc import ABCMeta
from pathlib import Path
from typing import Any

import joblib

from minst_digit_recognition.models.base import MlModel


class JoblibModelSerializationMixin(MlModel, metaclass=ABCMeta):
    """
    Mixin for MlModels serializable with joblib
    """
    _model: Any

    def load(self) -> None:
        self._model = joblib.load(self._get_model_filename())

    def save(self) -> None:
        self._check_initialized()
        joblib.dump(self._model, self._get_model_filename())

    def _get_model_filename(self):
        return Path(self.models_dir, f"{self.model_name}.joblib")
