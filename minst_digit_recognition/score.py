"""
executable package that create, fits and saves all available models
"""


def main():
    from minst_digit_recognition.data import TrainingData
    from minst_digit_recognition.models import get_models

    data = TrainingData()

    print("Scoring accuracy of models...")
    for _, model in get_models().items():
        print("Score", model.name, end=": ")
        model.load()
        print(model.evaluate_score(data))


if __name__ == "__main__":
    main()
