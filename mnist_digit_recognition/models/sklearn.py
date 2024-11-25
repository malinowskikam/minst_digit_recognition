"""
package containging sklearn based models
"""
import abc
import pathlib
from typing import Any
from functools import partial

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from mnist_digit_recognition.data import TrainingData
from mnist_digit_recognition.models.base import MlModel


class SkLearnClassifierModel(MlModel, metaclass=abc.ABCMeta):
    """
    Base Class for all sklearn based classifier models
    """

    _model: Any

    @property
    @abc.abstractmethod
    def classifier_class(self):
        pass

    @property
    @abc.abstractmethod
    def model_name(self):
        pass

    def __init__(self, model_name=None):
        super().__init__(model_name=model_name or self.model_name)

    def fit(self, data: TrainingData) -> None:
        self._model = self.classifier_class()
        self._model.fit(data.train_x, data.train_y)

    def predict(self, data: TrainingData):
        self._check_initialized()
        return self._model.predict(data.test_x)

    def evaluate_confusion_matrix(self, data: TrainingData):
        self._check_initialized()
        prediction = self.predict(data)
        return confusion_matrix(data.test_y, prediction, labels=list(range(10)))

    def evaluate_score(self, data: TrainingData):
        return round(self._model.score(data.test_x, data.test_y), 4)

    def load(self) -> None:
        self._model = joblib.load(self._get_model_filename())

    def save(self) -> None:
        self._check_initialized()
        joblib.dump(self._model, self._get_model_filename())

    def _get_model_filename(self):
        return pathlib.Path(self.models_dir, f"{self.name}.joblib")

    def ready(self) -> bool:
        return hasattr(self, "_model") and self._model

    def _check_initialized(self):
        if not self.ready():
            raise AttributeError("Model object not initiated. Load or fit the model first.")


class DecisionTree(SkLearnClassifierModel):
    classifier_class = DecisionTreeClassifier
    model_name = "decision_tree"


class NaiveBayes(SkLearnClassifierModel):
    classifier_class = GaussianNB
    model_name = "naive_bayes"


class RandomForest(SkLearnClassifierModel):
    classifier_class = partial(RandomForestClassifier, n_jobs=-1, n_estimators=20, random_state=234263)
    model_name = "random_forest"


class NearestNeighbours5(SkLearnClassifierModel):
    classifier_class = partial(KNeighborsClassifier, n_neighbors=5)
    model_name = "nearest_neighbours_5"


class SupportVector(SkLearnClassifierModel):
    classifier_class = SVC
    model_name = "support_vector"
