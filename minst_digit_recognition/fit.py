"""
executable package that create, fits and saves all available models
"""

if __name__ == "__main__":
    from minst_digit_recognition.data import TrainingData
    from minst_digit_recognition.models import get_models

    data = TrainingData()

    for _, model in get_models().items():
        print("Fitting:", model.name)
        model.fit(data)
        model.save()
