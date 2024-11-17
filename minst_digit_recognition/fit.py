"""
Executable package that create, fits and saves all available models
"""


def main():
    from minst_digit_recognition.data import TrainingData
    from minst_digit_recognition.models import get_models

    data = TrainingData()

    print("Fitting models...")
    for _, model in get_models().items():
        print("Fitting", model.name)
        model.fit(data)
        model.save()


if __name__ == "__main__":
    main()
