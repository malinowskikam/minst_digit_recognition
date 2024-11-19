"""
Executable package that scores all available models
Requires models to be loadable
"""


def main():
    from mnist_digit_recognition.data import TrainingData
    from mnist_digit_recognition.models import get_models

    data = TrainingData()

    print("Scoring accuracy of models...")
    for _, model in get_models().items():
        print("Score", model.name, end=": ")
        model.load()
        print(model.evaluate_score(data))


if __name__ == "__main__":
    main()
