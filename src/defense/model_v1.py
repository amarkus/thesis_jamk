from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    BatchNormalization,
)


class SequentialModelV1:

    def __init__(self):
        self.img_shape = (64, 64, 3)
        self.input_layer = Input(shape=self.img_shape, name="Input layer")
        self.encoder = Sequential(
            [
                Input(shape=self.img_shape),
                Conv2D(
                    128,
                    (3, 3),
                    padding="same",
                    activation="relu",
                    input_shape=self.img_shape,
                ),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), padding="same"),
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                BatchNormalization(),
                MaxPooling2D(pool_size=(2, 2), padding="same"),
            ],
            name="Encoder",
        )
        self.decoder = Sequential(
            [
                Conv2D(64, (3, 3), activation="relu", padding="same"),
                UpSampling2D((2, 2)),
                Conv2D(128, (3, 3), activation="relu", padding="same"),
                UpSampling2D((2, 2)),
                Conv2D(3, (3, 3), activation="sigmoid", padding="same"),
            ],
            name="Decoder",
        )
