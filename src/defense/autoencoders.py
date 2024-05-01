from keras.models import Model
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import EarlyStopping, CSVLogger
from keras.utils.vis_utils import plot_model
from src.lib.filesystemhelper import FileSystemHelper
import matplotlib.pyplot as plt
import os
import time


# Class Autoencoder (convolutional autoencoder)
class Autoencoder:
    """
    A class used to represent an Convolutional Autoencoder
    Takes tf.keras.model structure class as parameter
    so that different structures and their performance can be evaluated.
    ...

    Attributes
    ----------
    autoencoder_model : tf.keras.Model
        a convolutional autoencoder model used by this class in training
    verbose : bool
        Boolean flag used to indicate the need to print more logging/debugging info
    """

    def __init__(self, autoencoder_model, verbose=False):

        # Common properties
        self.start_time = time.strftime("%Y%m%d-%H%M%S")
        self.fshelper = FileSystemHelper()
        self.working_directory = self.fshelper.current_working_directory()
        self.run_results_dir = "src/run_results"
        self.autoencoder_model_name = str(type(autoencoder_model).__name__)

        # Model properties
        self.input_layer = autoencoder_model.input_layer
        self.encoder = autoencoder_model.encoder
        self.decoder = autoencoder_model.decoder
        self.autoencoder_model = self.build_model()

        # Compile model
        self.optimizer = Adam(learning_rate=0.001)
        self.loss = MeanSquaredError(name="mean_squared_error")
        self.model_metrics = ["accuracy", "mse"]
        self.autoencoder_model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=self.model_metrics
        )
        self.autoencoder_model.summary(expand_nested=True)

    def build_model(self):
        """Returns a convolutional autoencoder Model class combining input layer, encoder and decoder"""
        # Input Layer
        input_layer = self.input_layer
        # Encoder (Sequential) Layer
        encoded = self.encoder(input_layer)
        # Decoder (Sequential) Layer
        decoded = self.decoder(encoded)
        # Return Autoencoder model (encoder + decoder)
        return Model(input_layer, decoded)

    def train_model(
        self,
        train_generator,
        validation_generator,
        epochs,
        steps_per_epoch=20,
        batch_size=20,
        validation_steps=50,
        validation_freq=1,
        filename="model_autoencoder_thesis",
        run_results_directory="src/run_results",
    ):

        early_stopping = EarlyStopping(
            monitor="val_loss", min_delta=0, patience=30, verbose=1, mode="auto"
        )
        run_specific_subdir = (
            f"{self.autoencoder_model_name}_{epochs}-epoch_{filename}_{self.start_time}"
        )
        fshelper = FileSystemHelper()
        target_directory = fshelper.get_dir_path(
            self.working_directory,
            os.path.join(run_results_directory, run_specific_subdir),
        )
        self.run_results_dir = target_directory
        self.fshelper.create_directory(self.run_results_dir)
        csv_logger = CSVLogger(
            f"{self.run_results_dir}/log_run.csv",
            append=True,
            separator=";",
        )

        # Write model summary into file also
        self.autoencoder_model.summary(
            expand_nested=True, print_fn=self.model_summary_to_file
        )

        # Store screenshots of architecture before training starts
        self.save_model_architecture_as_image()

        # Fit the model.
        history = self.autoencoder_model.fit(
            train_generator,
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            validation_freq=validation_freq,
            callbacks=[early_stopping, csv_logger],
            verbose=1,
        )
        self.save_model()
        self.plot_training_progress(history)
        self.plot_model_accuracy(history)

    def save_model(self, model_directory="models", filename="anomalydetector"):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        target_directory = os.path.join(self.run_results_dir, model_directory)
        self.fshelper.create_directory(target_directory)

        model_weights_file = os.path.join(
            target_directory, f"weights_{filename}_{timestr}.h5"
        )
        self.autoencoder_model.save_weights(model_weights_file)

        model_file = os.path.join(target_directory, f"model_{filename}_{timestr}.keras")
        self.autoencoder_model.save(model_file)

    def model_summary_to_file(self, s, filename="modelsummary.txt"):
        summaryfile = f"{self.run_results_dir}/{filename}"
        with open(summaryfile, "a", encoding="UTF8") as f:
            print(s, file=f)

    def save_model_architecture_as_image(self, screenshots_directory="screenshots"):
        run_results_dir = (
            self.run_results_dir if hasattr(self, "run_results_dir") else "src"
        )
        target_directory = os.path.join(run_results_dir, screenshots_directory)
        self.fshelper.create_directory(target_directory)

        plot_model(
            self.autoencoder_model,
            to_file=os.path.join(target_directory, "model_plot.png"),
            show_shapes=True,
            show_layer_names=True,
        )
        plot_model(
            self.encoder,
            to_file=os.path.join(target_directory, "model_plot_encoder.png"),
            show_shapes=True,
            show_layer_names=True,
        )
        plot_model(
            self.decoder,
            to_file=os.path.join(target_directory, "model_plot_decoder.png"),
            show_shapes=True,
            show_layer_names=True,
        )

    def plot_training_progress(self, history, screenshots_directory="screenshots"):
        target_directory = os.path.join(self.run_results_dir, screenshots_directory)
        self.fshelper.create_directory(target_directory)

        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Training & validation loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Training loss", "Validation loss"], loc="upper right")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plot_to_file = os.path.join(
            target_directory, f"model_training_and_validation_loss_{timestr}.png"
        )
        plt.savefig(plot_to_file)
        plt.show()

    def plot_model_accuracy(self, history, screenshots_directory="screenshots"):
        target_directory = os.path.join(self.run_results_dir, screenshots_directory)
        self.fshelper.create_directory(target_directory)

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["training", "validation"], loc="upper left")
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plot_to_file = os.path.join(target_directory, f"model_accuracy_{timestr}.png")
        plt.savefig(plot_to_file)
        plt.show()

    def plot_comparison_image_pair(
        self,
        original,
        predicted,
        reconstruction_error,
        screenshots_directory="screenshots",
    ):
        fig, (img1, img2) = plt.subplots(1, 2)
        # Plots
        img1.imshow(original)
        img2.imshow(predicted)
        # Title
        fig.suptitle("Evaluate reconstruction")
        # Titles for each plot
        img1.set_title("Original image")
        img1.set_xlabel("Reconstruction error:")
        img2.set_title("Reconstructed image")

        img2.set_xlabel(reconstruction_error)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        target_directory = os.path.join(self.run_results_dir, screenshots_directory)
        self.fshelper.create_directory(target_directory)
        plot_to_file = os.path.join(
            target_directory, f"image_pair_comparison_{timestr}.png"
        )
        plt.savefig(plot_to_file)
        plt.show()

    def eval_model(self, test_generator, batch_size=128, model=None):
        # Add option to use existing model for evaluation
        model_to_evaluate = self.autoencoder_model if model is None else model

        # Evaluate the model on the test data using `evaluate`
        print("Evaluate on test data")
        score = model_to_evaluate.evaluate(
            test_generator, batch_size=batch_size, verbose=0
        )
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        print("Generate predictions for 1 sample")
        predictions = model_to_evaluate.predict(test_generator, steps=1)
        batch = next(test_generator)
        reconstruction_error = model_to_evaluate.evaluate(
            predictions, batch[0], batch_size=1
        )[0]
        img1 = batch[0][0]
        self.plot_comparison_image_pair(img1, predictions[0], reconstruction_error)
