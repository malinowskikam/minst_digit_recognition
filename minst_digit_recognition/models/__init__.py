from minst_digit_recognition.models.base import MlModel
from minst_digit_recognition.models.dt import DecisionTree


def get_models() -> dict[str, MlModel]:
    return {
        "dt": DecisionTree()
    }
