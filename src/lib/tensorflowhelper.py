import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D


class TensorFlowHelper:
    def predict(self, img, modelfile):
        model = keras.models.load_model(modelfile)

        return model.predict([[img]])

    def evaluate(self, reconstruction, img, modelfile):
        model = keras.models.load_model(modelfile)
        reconstruction_error = model.evaluate([reconstruction], [[img]], batch_size=1)[
            0
        ]

        return reconstruction_error

    def predict2(self, img, model):
        return model.predict([[img]])

    def evaluate2(reconstruction, img, model):
        reconstruction_error = model.evaluate([reconstruction], [[img]], batch_size=1)[
            0
        ]

        return reconstruction_error

    def create_ok_model(self, SIZE):
        model = Sequential()
        model.add(
            Conv2D(
                64,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(SIZE, SIZE, 3),
            )
        )
        model.add(MaxPooling2D((2, 2), padding="same"))
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), padding="same"))
        model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2), padding="same"))

        # Decoder
        model.add(Conv2D(16, (3, 3), activation="relu", padding="same"))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(UpSampling2D((2, 2)))

        model.add(Conv2D(3, (3, 3), activation="sigmoid", padding="same"))

        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
        model.summary()

        return model
