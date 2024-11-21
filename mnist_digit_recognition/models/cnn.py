import pathlib

from keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator, np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from mnist_digit_recognition.data import TrainingData
from mnist_digit_recognition.models.base import MlModel


class CnnModel(MlModel):
    _model: Sequential
    _optimizer: RMSprop
    _data_gen: ImageDataGenerator
    _lr_reduction: ReduceLROnPlateau

    def __init__(self, model_name="cnn", batch_size=86, epochs=15):
        super().__init__(model_name=model_name)
        self.batch_size = batch_size
        self.epochs = epochs
        self._model_path = pathlib.Path(self.models_dir, f"{self.name}_{self.epochs}_{self.batch_size}.keras")

    def load(self) -> None:
        self._model = keras_load_model(self._model_path)

    def save(self) -> None:
        self._check_initialized()
        self._model.save(self._model_path)

    def fit(self, data: TrainingData) -> None:
        self._create_model()

        train_x = (data.train_x / 255.0).reshape(-1, 28, 28, 1)
        test_x = (data.test_x / 255.0).reshape(-1, 28, 28, 1)

        train_y = to_categorical(data.train_y,num_classes=10)
        test_y = to_categorical(data.test_y, num_classes=10)

        self._data_gen.fit(train_x)

        dataset = self._data_gen.flow(
                train_x,
                train_y,
                batch_size=self.batch_size
            )

        self._model.fit(
            dataset,
            epochs=self.epochs,
            validation_data=(test_x, test_y),
            verbose=0,
            # steps_per_epoch=dataset.samples // self.batch_size,
            callbacks=[self._lr_reduction])

    def predict(self, data: TrainingData):
        self._check_initialized()
        test_x = (data.test_x / 255.0).reshape(-1, 28, 28, 1)
        return np.argmax(self._model.predict(test_x), axis=1)

    def evaluate_confusion_matrix(self, data: TrainingData):
        self._check_initialized()
        return confusion_matrix(data.test_y, self.predict(data), labels=list(range(10)))

    def evaluate_score(self, data: TrainingData):
        self._check_initialized()
        test_x = (data.test_x / 255.0).reshape(-1, 28, 28, 1)
        test_y = to_categorical(data.test_y, num_classes=10)
        return self._model.evaluate(test_x, test_y, verbose=0)[1]

    def _check_initialized(self) -> None:
        if not hasattr(self, "_model") or not self._model:
            raise AttributeError("Model object not initiated. Load or fit the model first.")

    def _create_model(self):
        self._model = Sequential()

        self._model.add(Input(shape=(28, 28, 1)))

        self._model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
        self._model.add(MaxPool2D(pool_size=(2, 2)))
        self._model.add(Dropout(0.25))

        self._model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
        self._model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        self._model.add(Dropout(0.25))

        self._model.add(Flatten())
        self._model.add(Dense(128, activation="relu"))
        self._model.add(Dropout(0.5))

        self._model.add(Dense(10, activation="softmax"))

        self._optimizer = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08)
        self._model.compile(optimizer=self._optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        self._data_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False
        )

        self._lr_reduction = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=3,
            verbose=1,
            factor=0.5,
            min_lr=0.00001
        )
