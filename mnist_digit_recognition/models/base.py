"""
package containing base machine learning model class
"""
import abc

from mnist_digit_recognition.data import InputData


class MlModel(metaclass=abc.ABCMeta):
    """
    Base machine learning model class
    """
    models_dir: str
    data_dir: str
    name: str

    def __init__(self,
                 model_name,
                 models_dir="models",
                 data_dir="data"):
        self.name = model_name
        self.models_dir = models_dir
        self.data_dir = data_dir

    @abc.abstractmethod
    def load(self) -> None:
        """
        Load model from disk. This method should make the model "ready", similar to fit.
        """
        pass

    @abc.abstractmethod
    def save(self) -> None:
        """
        Save model to disk
        """
        pass

    @abc.abstractmethod
    def fit(self, data: InputData) -> None:
        """
        Train model on provided data. This method should make the model "ready", similar to load.

        :param data: InputData object containing data compatible with this model
        """
        pass

    @abc.abstractmethod
    def predict(self, data: InputData):
        pass

    @abc.abstractmethod
    def evaluate_confusion_matrix(self, data: InputData):
        pass

    @abc.abstractmethod
    def evaluate_score(self, data: InputData):
        pass

    @abc.abstractmethod
    def ready(self) -> bool:
        """
        Check if current model instance is loaded/fitted

        :return: True if model is ready, false otherwise
        """
        pass
