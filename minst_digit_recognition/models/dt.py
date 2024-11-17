"""
package containing decision tree ml model
"""
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from minst_digit_recognition.data import TrainingData
from minst_digit_recognition.models.base import MlModel, JoblibModelSerializationMixin


class DecisionTree(JoblibModelSerializationMixin, MlModel):
    """
    Decision Tree classifier implemented with sklearn
    """

    _model: DecisionTreeClassifier

    def __init__(self, model_name="decision_tree"):
        super().__init__(model_name=model_name)

    def fit(self, data: TrainingData) -> None:
        self._model = DecisionTreeClassifier()
        self._model.fit(data.train_x, data.train_y)

    def predict(self, data: TrainingData):
        self._check_initialized()
        return self._model.predict(data.test_x)

    def evaluate_confusion_matrix(self, data: TrainingData):
        self._check_initialized()
        prediction = self.predict(data)
        return confusion_matrix(data.test_y, prediction, labels=list(range(10)))

    def evaluate_score(self, data: TrainingData):
        return self._model.score(data.test_x, data.test_y)

    def _check_initialized(self):
        if not hasattr(self, "_model") or not self._model:
            raise AttributeError("Model object not initiated. Load or fit the model first.")
