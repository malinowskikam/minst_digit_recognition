from keras import Sequential
from keras.callbacks import ReduceLROnPlateau
from keras.engine.saving import load_model
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator

batch_size = 86
epochs = 10
model_dir = "models\\"

name = "convolutional_neural_network"

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
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

learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy', patience=3,
    verbose=1,
    factor=0.5,
    min_lr=0.00001
)


def fit_nn(data):
    datagen.fit(data[0])
    history = model.fit_generator(
        datagen.flow(
            data[0],
            data[2],
            batch_size=batch_size
        ),
        epochs=epochs,
        validation_data=(data[1], data[3]),
        verbose=2,
        steps_per_epoch=data[0].shape[0] // batch_size,
        callbacks=[learning_rate_reduction])

    model.save(model_dir + name + "_" + str(epochs) + "_" + str(batch_size) + ".h5")

    return history

def predict_nn(data):
    return model.predict(data)
