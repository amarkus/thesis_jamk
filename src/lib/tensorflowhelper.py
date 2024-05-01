from tensorflow import keras


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
