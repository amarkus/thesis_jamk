from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from lib.filesystemhelper import FileSystemHelper
from lib.csvhelper import CsvHelper
from lib.httphelper import HttpHelper
from lib.tensorflowhelper import TensorFlowHelper

# Load helpers
filesystemhelper = FileSystemHelper()
csvhelper = CsvHelper()
httphelper = HttpHelper()
tfhelper = TensorFlowHelper()

# Define path used
working_directory = filesystemhelper.current_working_directory()
basepath = os.path.normpath(os.path.join(working_directory, "src"))
evaluation_path = os.path.join(basepath, "dataset/evaluation_patches")
model_directory = os.path.join(basepath, "models")

# Load autoencoder
autoencoder_model_name = "model_autoencoder_thesis.keras"
autoencoder_model_path = os.path.join(model_directory, autoencoder_model_name)
model = keras.models.load_model(autoencoder_model_path)

# CSV columns:
fieldnames = ["image_name", "reconstruction_error", "actual_class", "predicted_class"]

# Value found using different methods in: find_anomaly_threshold.py
threshold = 0.000554


def compute_reconstruction_error_for_image(img_path):
    img = filesystemhelper.read_image_as_array(img_path)
    reconstruction = tfhelper.predict(img, autoencoder_model_path)
    reconstruction_error = tfhelper.evaluate(
        reconstruction, img, autoencoder_model_path
    )

    return reconstruction_error


def is_anomaly_image(img_path):
    if "_anomaly.png" in img_path:
        return True
    return False


def get_class_name(is_anomaly):
    labels = ["Normal", "Anomaly"]
    if is_anomaly:
        return labels[1]
    return labels[0]


def get_predicted_class(prediction):
    result_str = "prediction: {}, treshold: {}".format(prediction, threshold)
    print(result_str)

    if prediction <= threshold:
        print("normal")
        return 0
    print("anomaly")
    return 1


def bool_to_int(x):
    if x:
        return 1
    return 0


def int_to_bool(is_anomaly):
    if is_anomaly:
        return True
    return False


def print_and_save_classification_report(truth, prediction):
    print(classification_report(truth, prediction))
    report = classification_report(truth, prediction)
    report_path = "classification_report.txt"
    text_file = open(report_path, "w")
    n = text_file.write(report)
    text_file.close()


def get_reconstruction_errors(path, output_filename, count: int = 10):
    print(path)
    image_paths = filesystemhelper.read_images_from_folder(
        path, count, recursive_subfolders=True
    )
    rows = []
    actual_classes = []
    predictions = []
    for img_path in image_paths:
        print("img_path:", img_path)
        image_name = os.path.basename(img_path)
        reconstruction_error = compute_reconstruction_error_for_image(img_path)
        is_anomaly = is_anomaly_image(img_path)
        actual_class_as_number = bool_to_int(is_anomaly)
        actual_classes.append(actual_class_as_number)
        actual_class = get_class_name(is_anomaly)
        print("actual_class:", actual_class)
        print("actual_class_as_number:", actual_class_as_number)
        predicted_class_as_number = get_predicted_class(reconstruction_error)
        prediction_is_anomaly = int_to_bool(predicted_class_as_number)
        predicted_class = get_class_name(prediction_is_anomaly)
        print("predicted_class:", predicted_class)
        print("predicted_class_as_number:", predicted_class_as_number)
        predictions.append(predicted_class_as_number)
        row = {
            "image_name": image_name,
            "reconstruction_error": reconstruction_error,
            "actual_class": actual_class,
            "predicted_class": predicted_class,
        }
        rows.append(row)

    csvhelper.write_data_to_csv(output_filename, fieldnames, rows)
    return (actual_classes, predictions)


# Save evaluation results
(actuals, predictions) = get_reconstruction_errors(
    evaluation_path, "evaluation_set_results.csv", 200
)


# Plot confusion matrix
truth = np.array(actuals)
prediction = np.array(predictions)
labels = ["Normal", "Anomaly"]

print("actuals    :", actuals)
print("predictions:", predictions)

cm = confusion_matrix(truth, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix.png")
plt.show()

# Print classification report
print_and_save_classification_report(truth, prediction)
