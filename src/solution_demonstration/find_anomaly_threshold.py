# Run comparison tests against truth results & predicted results
# Select an optimal anomaly threshold based on:
#   - mean value of (the mean values) both original and anomaly datasets
#   - or based on: F1, recall & precision. What ever works the best when tested :)
import os
import numpy as np
import matplotlib.pyplot as plt
import texttable
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from src.lib.filesystemhelper import FileSystemHelper
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


SIZE = 64
batch_size = 64
datagen = ImageDataGenerator(rescale=1.0 / 255)
fshelper = FileSystemHelper()
working_directory = fshelper.current_working_directory()
basepath = os.path.join(working_directory, "src")
validation_path = os.path.normpath(
    os.path.join(basepath, "dataset/demonstration_patches")
)
anomaly_path = os.path.normpath(os.path.join(basepath, "dataset/demonstration_patches"))

validation_generator = datagen.flow_from_directory(
    validation_path,
    classes=["original"],
    target_size=(SIZE, SIZE),
    batch_size=100,
    class_mode="input",
    subset=None,
    shuffle=False,
    seed=1,
)

anomaly_generator = datagen.flow_from_directory(
    anomaly_path,
    classes=["adversarial"],
    target_size=(SIZE, SIZE),
    batch_size=100,
    class_mode="input",
    subset=None,
    shuffle=False,
    seed=1,
)

# Load autoencoder model
model_path = os.path.normpath(
    os.path.join(basepath, "models/model_autoencoder_thesis.keras")
)
model = keras.models.load_model(model_path)


def get_true_value(filename=""):
    # Return 1 for anomaly and 0 for normal image
    if "_anomaly" in filename:
        return 1
    return 0


def get_true_values(filenames):
    labels = []
    for file in filenames:
        l = get_true_value(file)
        labels.append(l)
    return labels


def get_file_name(idg=None, index: int = 10):
    filename = idg.filenames[index]
    image_name = os.path.basename(filename)
    print(image_name)
    return image_name


def get_file_name_from_data(idg, count: int = 10):
    filename = idg.filenames
    result_filenames = []
    for f in range(0, count):
        print(filename[f])
        result_filenames.append(filename[f])
    return result_filenames


def load_sample_list(image_data_generator, count: int = 10):
    data_list = []
    batch_index = 0
    while batch_index <= count:
        data = image_data_generator.next()
        data_list.append(data[0])
        batch_index = batch_index + 1

    return data_list


def predict_image(image):
    img = image[np.newaxis, :, :, :]
    reconstructed_img = model.predict([[img]])
    plt.imshow(reconstructed_img[0])
    plt.show()
    return reconstructed_img


def compute_recon_error_metrics(batch_images, size=64):
    rec_error_list = []
    counter = 0
    # batch_size = batch_images.shape[0] - 1

    # for i in range(0, batch_images.shape[0] - 1):
    for i in range(0, size):
        img = batch_images[i]
        img = img[np.newaxis, :, :, :]
        rec = model.predict([[img]])
        rec_error = model.evaluate([rec], [[img]], batch_size=1)[0]
        rec_error_list.append(rec_error)
        counter += 1

    average_rec_error = np.mean(np.array(rec_error_list))
    stdev_rec_error = np.std(np.array(rec_error_list))

    return (average_rec_error, stdev_rec_error, rec_error_list)


def predict_from_threshold(threshold=0.5, arr_input=[]):
    arr_predicted = []
    for val in arr_input:
        if val >= threshold:
            arr_predicted.append(1)
        else:
            arr_predicted.append(0)

    return arr_predicted


def print_threshold_results(acc, f1, recall, precision):
    # Create texttable object
    tableObj = texttable.Texttable()
    tableObj._precision = 8
    # Set columns
    tableObj.set_cols_align(["l", "l", "l", "l", "l"])
    # Set datatype of each column
    tableObj.set_cols_dtype(["f", "f", "f", "f", "f"])
    # Insert rows
    tableObj.add_rows(
        [
            ["Threshold", "Accuracy", "F1 score", "Recall", "Precision"],
        ]
    )
    tableObj.add_rows([[0, 0, 0, 0, 0]])
    str = tableObj.draw()


def find_optimal_threshold(
    truth_values,
    predicted_values,
    start=0.0004,
    stop=0.0006,
    step=0.000005,
):
    scorings = dict()

    print("-------------------------------------------------------------")
    for th in np.arange(start, stop, step):
        predicted = predict_from_threshold(threshold=th, arr_input=predicted_values)
        precision, recall, fscore, support = precision_recall_fscore_support(
            truth_values, predicted, average="macro"
        )
        f1 = f1_score(truth_values, predicted, zero_division=0)
        scorings[th] = f1
        print("Threshold:", th)
        print("F1 score:", f1)
        print("-------------------------------------------------------------")

    max_key = max(scorings, key=lambda x: scorings[x])
    best_threshold = max_key
    best_f1_score = scorings[best_threshold]
    return (best_threshold, best_f1_score)


validation_batch = validation_generator.next()[0]
anomaly_batch = anomaly_generator.next()[0]

(val_avg_rec_error, val_stdev_rec_error, val_errors) = compute_recon_error_metrics(
    validation_batch, 100
)
(anomaly_avg_rec_error, anomaly_stdev_rec_error, anomaly_errors) = (
    compute_recon_error_metrics(anomaly_batch, 100)
)

combined_predicted_values = val_errors + anomaly_errors
print(combined_predicted_values)
print(len(combined_predicted_values))

normal_img_paths = get_file_name_from_data(validation_generator, 100)
normal_img_truth = get_true_values(normal_img_paths)
anomaly_img_paths = get_file_name_from_data(anomaly_generator, 100)
anomaly_img_truth = get_true_values(anomaly_img_paths)
combined_truth_values = normal_img_truth + anomaly_img_truth
print(normal_img_truth)
print(anomaly_img_truth)
print(combined_truth_values)
print(len(combined_truth_values))


# Combine validation & anomaly reconstruction error lists
arr_normal_and_anomaly = [val_avg_rec_error, anomaly_avg_rec_error]
# Calculate simply the mean of both error averages(means)
mean_of_averages = np.mean(arr_normal_and_anomaly)

# Print calculation results
print(f"Average (mean) recognition error of validation images: {val_avg_rec_error}")
print(f"Standard deviation rec. error of validation images: {val_stdev_rec_error}")
print(f"Average (mean) recognition error of anomaly images: {anomaly_avg_rec_error}")
print(f"Standard deviation rec. error of anomaly images: {anomaly_stdev_rec_error}")
print(f"Good threshold value: {mean_of_averages}")

""" normal_sample_list = load_sample_list(validation_generator, 5)
anomaly_sample_list = load_sample_list(anomaly_generator, 5)
first_batch_normal = normal_sample_list[0] """
predicted = predict_from_threshold(
    threshold=0.0005, arr_input=combined_predicted_values
)
print(combined_truth_values)
print(predicted)

precision, recall, fscore, support = precision_recall_fscore_support(
    combined_truth_values, predicted, average="macro"
)

print("Accuracy:", accuracy_score(combined_truth_values, predicted))
print("F1 score:", f1_score(combined_truth_values, predicted, zero_division=0))
print("Recall:", recall_score(combined_truth_values, predicted, zero_division=0))
print("Precision:", precision_score(combined_truth_values, predicted, zero_division=0))
print(
    "\n clasification report:\n",
    classification_report(combined_truth_values, predicted, zero_division=0),
)

(best_threshold, best_f1_score) = find_optimal_threshold(
    start=val_avg_rec_error,
    stop=anomaly_avg_rec_error,
    step=0.000005,
    truth_values=combined_truth_values,
    predicted_values=combined_predicted_values,
)

print("best_threshold:", best_threshold)
print("best_f1_score:", best_f1_score)


# conf matrix
truth = np.array(combined_truth_values)
prediction = np.array(predicted)
labels = ["Normal", "Anomaly"]

print("actuals    :", combined_truth_values)
print("predictions:", predicted)

cm = confusion_matrix(truth, prediction)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues)
plt.savefig("confusion_matrix_threshold.png")
plt.show()
