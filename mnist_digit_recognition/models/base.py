"""
package containing base machine learning model class
"""
import abc

from mnist_digit_recognition.data.trainingData import TrainingData


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
        Load model from disk
        """
        pass

    @abc.abstractmethod
    def save(self) -> None:
        """
        Save model to disk
        """
        pass

    @abc.abstractmethod
    def fit(self, data: TrainingData) -> None:
        """
        Train model on provided data

        :param data: TrainingData object containing data compatible with this model
        """
        pass

    @abc.abstractmethod
    def predict(self, data: TrainingData):
        pass

    @abc.abstractmethod
    def evaluate_confusion_matrix(self, data: TrainingData):
        pass

    @abc.abstractmethod
    def evaluate_score(self, data: TrainingData):
        pass

    @abc.abstractmethod
    def _check_initialized(self) -> None:
        """
        Check if the current instance of MlModel is initialized
        Should rise attribute error if not initialized
        """
        pass