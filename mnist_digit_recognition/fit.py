"""
Executable package that create, fits and saves all available models
"""


def main():
    from mnist_digit_recognition.data import InputData
    from mnist_digit_recognition.models import get_models

    data = InputData()
    data.load()

    print("Fitting models...")
    for _, model in get_models().items():
        print("Fitting", model.name)
        model.fit(data)
        model.save()


if __name__ == "__main__":
    main()
