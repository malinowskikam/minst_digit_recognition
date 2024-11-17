"""
Executable package that evaluates confusion matrix for all available models
Requires models to be loadable
"""


def main():
    from minst_digit_recognition.data import TrainingData
    from minst_digit_recognition.models import get_models

    data = TrainingData()

    print("Evaluating confusion matrices for models...")
    for _, model in get_models().items():
        model.load()
        print(model.name)
        print(model.evaluate_confusion_matrix(data))


if __name__ == "__main__":
    main()
