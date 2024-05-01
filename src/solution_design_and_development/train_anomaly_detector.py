from src.preprocess_dataset.dataset import MitosisDataset
from defense.autoencoders import Autoencoder
from defense.model_v1 import SequentialModelV1
from defense.model_v2 import SequentialModelV2
import os

# Define model architecture to use
convolutional_autoencoder_model = SequentialModelV1()
anomaly_detector = Autoencoder(convolutional_autoencoder_model)

# Define datasets
mitosis_dataset = MitosisDataset(image_size=(64, 64), batch_size=64, verbose=False)
dataset_folder = os.path.normpath(
    os.path.join(mitosis_dataset.working_directory, "src/dataset/training_data")
)

# Load split datasets
train_ds = mitosis_dataset.load_training_data(directory=dataset_folder, augment=False)
val_ds = mitosis_dataset.load_validation_data(directory=dataset_folder)
test_ds = mitosis_dataset.load_test_data(directory=dataset_folder)

# Train using train & validation datasets and save the AI model
anomaly_detector.train_model(
    train_ds,
    val_ds,
    epochs=500,
    steps_per_epoch=None,
    batch_size=None,
    validation_steps=None,
    filename="autoencoder",
)

# Evaluate model performance against test dataset
anomaly_detector.eval_model(test_ds)
