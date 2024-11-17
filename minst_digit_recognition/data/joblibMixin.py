from pathlib import Path
from typing import Any

import joblib


class JoblibModelSerializationMixin:
    _model: Any
    _model_path: Path

    def load(self) -> None:
        self._model = joblib.load(self._model_path)

    def save(self) -> None:
        if not self._model:
            raise AttributeError("Model")
        joblib.dump(self._model, self._model_path)
