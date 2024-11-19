from mnist_digit_recognition.models.base import MlModel as _MlModel
from mnist_digit_recognition.models import sklearn as _sk


def get_models() -> dict[str, _MlModel]:
    return {
        "nb": _sk.NaiveBayes(),
        "dt": _sk.DecisionTree(),
        "rf": _sk.RandomForest(),
        "nn5": _sk.NearestNeighbours5(),
        "sv": _sk.SupportVector(),
    }
