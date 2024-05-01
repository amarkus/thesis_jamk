import os, time
import matplotlib.pyplot as plt
import texttable
from lib.filesystemhelper import FileSystemHelper
from lib.csvhelper import CsvHelper
from lib.httphelper import HttpHelper
from lib.tensorflowhelper import TensorFlowHelper
from tensorflow import keras
from functools import reduce
from statistics import mean, median, stdev


class Evaluation:

    def __init__(
        self,
        autoencoder_model_path="src/models",
        autoencoder_model_name="model_autoencoder.keras",
        evaluation_result_dir="src/reports/evaluation",
        normal_images_path="src/dataset/evaluation_patches/original",
        adversarial_images_path="src/dataset/evaluation_patches/adversarial",
    ):
        # Common properties
        self.start_time = time.strftime("%Y%m%d-%H%M%S")
        self.filesystemhelper = FileSystemHelper()
        self.csvhelper = CsvHelper()
        self.httphelper = HttpHelper()
        self.tfhelper = TensorFlowHelper()
        self.working_directory = self.filesystemhelper.current_working_directory()
        self.model_path = autoencoder_model_path
        self.results_dir = os.path.join(evaluation_result_dir, f"{self.start_time}")
        self.normal_images_dir = normal_images_path
        self.adversarial_images_dir = adversarial_images_path

        # Autoencoder model
        self.model_name = autoencoder_model_name
        autoencoder_model_path = os.path.join(self.model_path, autoencoder_model_name)
        self.autoencoder_model = keras.models.load_model(autoencoder_model_path)

        # Result file names
        self.filesystemhelper.create_directory(self.results_dir)
        self.normal_img_predictions_file = os.path.join(
            self.results_dir,
            f"normal_image_predictions.csv",
        )
        self.mitosis_img_predictions_file = os.path.join(
            self.results_dir,
            f"mitosis_image_predictions.csv",
        )
        self.all_img_predictions_file = os.path.join(
            self.results_dir,
            f"all_image_predictions.csv",
        )
        self.all_adversarial_img_predictions_file = os.path.join(
            self.results_dir,
            f"all_adversarial_image_predictions.csv",
        )
        self.prediction_summary_file = os.path.join(
            self.results_dir,
            f"predictions_summary.txt",
        )

        # csv header
        self.attack_evaluation_fieldnames = [
            "image_name",
            "initial_prediction",
            "prediction_after_attack",
            "initial_reconstruction_error",
            "reconstruction_error_after_attack",
        ]

        # Counters to evaluate performance
        # Normal images
        self.all_images_predictions = []
        self.normal_images_predictions = []
        self.mitosis_images_predictions = []
        self.all_images_reconstruction_error = []
        self.normal_images_reconstruction_error = []
        self.mitosis_images_reconstruction_error = []
        # Adversarial images
        self.all_adv_images_predictions = []
        self.adv_normal_images_predictions = []
        self.adv_mitosis_images_predictions = []
        self.all_adv_images_reconstruction_error = []
        self.adv_normal_images_reconstruction_error = []
        self.adv_mitosis_images_reconstruction_error = []

        # Rows to use for reports
        self.all_images_rows = []
        self.normal_images_rows = []
        self.mitosis_images_rows = []
        self.all_adv_images_rows = []
        self.adv_normal_images_rows = []
        self.adv_mitosis_images_rows = []

    def Average(self, lst):
        return reduce(lambda a, b: a + b, lst) / len(lst)

    def print_average_prediction_error(self, label, arr):
        average_predition_error = self.Average(arr)
        result_str = "average predition error for {}: {}".format(
            label, average_predition_error
        )
        print(result_str)

    def print_summary(
        self, before_data=[], after_data=[], title="", print_to_file=True
    ):
        # Values from the data before attack
        # "Autoencoder reconstruction error:{}".format("%.5f" % reconstruction_error)
        max_value_before = "{:.9f}".format(max(before_data))
        mean_value_before = "{:.9f}".format(mean(before_data))
        median_value_before = "{:.9f}".format(median(before_data))
        stdev_value_before = "{:.9f}".format(stdev(before_data))
        min_value_before = "{:.9f}".format(min(before_data))

        # Values from the data after attack
        max_value_after = "{:.9f}".format(max(after_data))
        mean_value_after = "{:.9f}".format(mean(after_data))
        median_value_after = "{:.9f}".format(median(after_data))
        stdev_value_after = "{:.9f}".format(stdev(after_data))
        min_value_after = "{:.9f}".format(min(after_data))

        # Create texttable object
        tableObj = texttable.Texttable()
        tableObj._precision = 8
        # Set columns
        tableObj.set_cols_align(["l", "r", "r"])
        # Set datatype of each column
        tableObj.set_cols_dtype(["t", "f", "f"])
        # Insert rows
        tableObj.add_rows(
            [
                ["", "Before attack", "After attack"],
                ["Maximum", max_value_before, max_value_after],
                ["Mean", mean_value_before, mean_value_after],
                ["Median", median_value_before, median_value_after],
                ["Standard deviation", stdev_value_before, stdev_value_after],
                ["Minimum", min_value_before, min_value_after],
            ]
        )
        str = tableObj.draw()
        print("+--------------------+---------------+--------------+")
        print(f"| {title} |")
        print(str)
        if print_to_file:
            with open(self.prediction_summary_file, "a", encoding="UTF8") as f:
                print("+--------------------+---------------+--------------+", file=f)
                print(f"| {title} |", file=f)
                print(str, file=f)
                print(" ", file=f)

    def boxplot_results(self, data1, data2, label1, label2, title="boxplot"):
        # Creating boxplot
        box_plot_file = os.path.join(
            self.results_dir,
            f"box_plot_{title}.png",
        )
        # fig = plt.figure(figsize=(10, 7))
        data = [data1, data2]
        plt.title(title)
        plt.grid(axis="y")
        plt.boxplot(
            data,
            labels=[label1, label2],
            showfliers=True,
            showmeans=True,
            meanline=True,
            notch=False,
            patch_artist=False,
        )

        plt.savefig(box_plot_file)
        plt.show()

    def histogram_results(self, data1, data2, label1, label2, title="histogram"):
        # Creating histogram
        histogram_file = os.path.join(
            self.results_dir,
            f"histogram_{title}.png",
        )

        plt.hist(data1, bins=15, alpha=0.5, label=label1)
        plt.hist(data2, bins=5, alpha=0.5, label=label2)
        plt.legend()
        plt.title(title)
        plt.xlabel("CODAIT prediction", size=11)
        plt.ylabel("Count", size=11)
        plt.savefig(histogram_file)
        plt.show()

    def get_cwd_relative_path(self, path):
        basepath = os.path.normpath(os.path.join(self.working_directory, path))
        return basepath

    def compute_reconstruction_error_for_image(self, img_path):
        img = self.filesystemhelper.read_image_as_array(img_path)
        reconstruction = self.autoencoder_model.predict([[img]])
        reconstruction_error = self.autoencoder_model.evaluate(
            [reconstruction], [[img]], batch_size=1
        )[0]

        return reconstruction_error

    # Compute mitosis image values:
    def codait_result_for_image(self, image):
        result = self.httphelper.post_file(image)
        print("Image", image, " has mitosis value: ", result)
        return result

    def codait_result_for_images(self, count: 10, recursive=False):
        basepath_normal = os.path.normpath(
            os.path.join(self.working_directory, self.normal_images_dir)
        )
        basepath_adversarial = os.path.normpath(
            os.path.join(self.working_directory, self.adversarial_images_dir)
        )
        normal_image_paths = self.filesystemhelper.read_images_from_folder(
            basepath_normal, count, recursive_subfolders=recursive
        )
        for image in normal_image_paths:
            # CODAIT prediction for original
            prediction_before = self.codait_result_for_image(image)
            reconstruction_error_before = self.compute_reconstruction_error_for_image(
                image
            )
            # CODAIT prediction for adversarial sample
            image_name_only = os.path.basename(image)
            name, ext = os.path.splitext(image_name_only)
            adv_image = os.path.join(basepath_adversarial, f"{name}_anomaly{ext}")
            prediction_after = self.codait_result_for_image(adv_image)
            reconstruction_error_after = self.compute_reconstruction_error_for_image(
                adv_image
            )

            self.add_to_results(
                image_name_only,
                prediction_before,
                prediction_after,
                reconstruction_error_before,
                reconstruction_error_after,
            )

    def add_to_results(
        self,
        image_name_only,
        prediction_before,
        prediction_after,
        reconstruction_error_before,
        reconstruction_error_after,
    ):
        row = {
            "image_name": image_name_only,
            "initial_prediction": prediction_before,
            "prediction_after_attack": prediction_after,
            "initial_reconstruction_error": reconstruction_error_before,
            "reconstruction_error_after_attack": reconstruction_error_after,
        }
        if "normal_" in image_name_only:
            self.normal_images_rows.append(row)
            self.normal_images_predictions.append(float(prediction_before))
            self.adv_normal_images_predictions.append(float(prediction_after))
            self.normal_images_reconstruction_error.append(
                float(reconstruction_error_before)
            )
            self.adv_normal_images_reconstruction_error.append(
                float(reconstruction_error_after)
            )
        if "mitosis_" in image_name_only:
            self.mitosis_images_rows.append(row)
            self.mitosis_images_predictions.append(float(prediction_before))
            self.adv_mitosis_images_predictions.append(float(prediction_after))
            self.mitosis_images_reconstruction_error.append(
                float(reconstruction_error_before)
            )
            self.adv_mitosis_images_reconstruction_error.append(
                float(reconstruction_error_after)
            )
        if "_anomaly" in image_name_only:
            self.all_adv_images_rows.append(row)
            self.all_adv_images_predictions.append(float(prediction_after))
            self.all_adv_images_reconstruction_error.append(
                float(reconstruction_error_after)
            )

        self.all_images_rows.append(row)
        self.all_images_predictions.append(float(prediction_before))
        self.all_images_reconstruction_error.append(float(reconstruction_error_before))
        self.all_adv_images_reconstruction_error.append(
            float(reconstruction_error_after)
        )

    def save_normal_images_as_csv(self):
        self.csvhelper.write_data_to_csv(
            self.normal_img_predictions_file,
            self.attack_evaluation_fieldnames,
            self.normal_images_rows,
        )

    def save_mitosis_images_as_csv(self):
        self.csvhelper.write_data_to_csv(
            self.mitosis_img_predictions_file,
            self.attack_evaluation_fieldnames,
            self.mitosis_images_rows,
        )

    def save_all_images_as_csv(self):
        self.csvhelper.write_data_to_csv(
            self.all_img_predictions_file,
            self.attack_evaluation_fieldnames,
            self.all_images_rows,
        )

    def save_adversarial_images_as_csv(self):
        self.csvhelper.write_data_to_csv(
            self.all_adversarial_img_predictions_file,
            self.attack_evaluation_fieldnames,
            self.all_adv_images_rows,
        )
