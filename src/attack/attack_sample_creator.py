import os
import numpy as np
import time
from pathlib import Path
from PIL import Image
from attack.pixel_attacks import FewPixelAttacker
from src.lib.filesystemhelper import FileSystemHelper


class AdversarialImagePatchCreator:

    def __init__(
        self,
        basepath="src/dataset",
        normal_images_path="original_patches/normal",
        mitosis_images_path="original_patches/mitosis",
        adversarial_images_path="adversarial_patches",
        attack_logs_path="src/reports/black_box_attacks",
        patch_type="normal",
        verbose=False,
    ):
        # Get current working directory path
        self.fshelper = FileSystemHelper()
        self.start_time = time.strftime("%Y%m%d-%H%M%S")
        self.patch_type = patch_type
        self.working_directory = self.fshelper.current_working_directory()
        self.basepath = os.path.normpath(os.path.join(self.working_directory, basepath))
        self.normal_images_path = self.fshelper.get_dir_path(
            self.basepath, normal_images_path
        )
        self.mitosis_images_path = self.fshelper.get_dir_path(
            self.basepath, mitosis_images_path
        )
        self.adversarial_images_path = self.fshelper.get_dir_path(
            self.basepath, adversarial_images_path
        )
        self.attack_logs_path = os.path.normpath(
            os.path.join(self.working_directory, attack_logs_path)
        )

        # Create output image directories if not exists
        self.fshelper.create_directory(self.adversarial_images_path)
        self.fshelper.create_directory(self.attack_logs_path)

        # Color bounds to be used in attack
        self.near_black_bounds = [(0, 5), (0, 5), (0, 5)]
        self.pink_bounds = [(220, 225), (170, 175), (200, 205)]

        # Additional color bounds for testing
        self.yellow_bounds = [(254, 255), (254, 255), (202, 204)]
        self.purple_bounds = [(95, 100), (52, 58), (125, 130)]

        # Maximum number of generations over which the entire population is evolved
        self.max_number_of_generations = 500

        # Multiplier for setting the total population size
        self.population_size = 30

        # Should the function be minimized? For mitosis yes, for normal images no.
        self.min_fn_for_mitosis = True
        self.min_fn_for_normal = False

        # Amount of change expected
        self.expected_change_factor_mitosis = 80
        self.expected_change_factor_normal = 80

        # Counters to evaluate performance
        self.time_taken = []
        self.number_of_http_requests = []
        self.prediction_change = []

        # Verbose logging
        self.verbose_logging = verbose

    # Treshold calculated using original prediction & expected change factor
    def mitosis_threshold_val(self, original_prediction):
        return float(original_prediction) / self.expected_change_factor_mitosis

    def mitosis_threshold_fn(self, original_prediction, prediction):
        print("prediction:", "%.10f" % prediction)
        success_threshold_value = self.mitosis_threshold_val(original_prediction)
        return float(prediction) < success_threshold_value

    # Treshold calculated using original prediction & expected change factor
    def normal_threshold_val(self, original_prediction):
        return float(original_prediction) * self.expected_change_factor_normal

    def normal_threshold_fn(self, original_prediction, prediction):
        print("prediction:", "%.10f" % prediction)
        success_threshold_value = 1 - self.normal_threshold_val(original_prediction)
        return prediction > success_threshold_value

    # Store adversarial image patch into adversarial patches folder
    def save_attacked_image(self, img_data, image_path):
        image = Image.fromarray(img_data).convert("RGB")
        image_name_only = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name_only)[0]
        image_ext = os.path.splitext(image_name_only)[1]
        image_modified_name_with_ext = "{}_anomaly{}".format(
            image_name_no_ext, image_ext
        )
        if self.verbose_logging:
            print("image_modified_name_with_ext:", image_modified_name_with_ext)
        attacked_image_path = os.path.join(
            self.adversarial_images_path, image_modified_name_with_ext
        )
        image.save(attacked_image_path)

    def store_single_attack_performance(self, httpRequests, timeTaken):
        self.number_of_http_requests.append(httpRequests)
        self.time_taken.append(timeTaken)

    def print_single_attack_performance(self, httpRequests, timeTaken):
        print("Number of HTTP requests used:", httpRequests)
        print("Time taken in seconds:", timeTaken)

    def average(self, list):
        return sum(list) / len(list)

    def overall_attack_performance(self):
        if len(self.number_of_http_requests) < 1:
            return

        avg_number_of_http_requests = self.average(self.number_of_http_requests)
        avg_time_taken = self.average(self.time_taken)
        avg_prediction_change = self.average(self.prediction_change)
        print("------------------------------------------------------")
        print("Average number of HTTP requests used:", avg_number_of_http_requests)
        print("Average time taken in seconds:", avg_time_taken)
        print("Average prediction change:", avg_prediction_change)
        print("------------------------------------------------------")

    def attack_performance_to_file(self):
        if len(self.number_of_http_requests) < 1:
            return

        avg_number_of_http_requests = self.average(self.number_of_http_requests)
        avg_time_taken = self.average(self.time_taken)
        avg_prediction_change = self.average(self.prediction_change)

        attack_log_file = os.path.join(
            self.attack_logs_path,
            f"black_box_attack_log_{self.patch_type}_{self.start_time}.txt",
        )

        with open(attack_log_file, "a", encoding="UTF8") as f:
            print("------------------------------------------------------", file=f)
            print(
                "Average number of HTTP requests used:",
                avg_number_of_http_requests,
                file=f,
            )
            print("Average time taken in seconds:", avg_time_taken, file=f)
            print("Average prediction change:", avg_prediction_change, file=f)
            print("------------------------------------------------------", file=f)

    def mitosis_perform_attack_and_get_result(self, img_data, image_path):
        few_pixel_attacker = FewPixelAttacker(
            dimensions=(64, 64), verbose=self.verbose_logging
        )
        print("----------[%s]----------" % Path(image_path).name)
        results = few_pixel_attacker.attack(
            img_data,
            pixel_count=5,
            maxiter=self.max_number_of_generations,
            popsize=self.population_size,
            threshold_fn=self.mitosis_threshold_fn,
            threshold_val=self.mitosis_threshold_val,
            minimize=self.min_fn_for_mitosis,
            color_bounds=self.pink_bounds,
        )
        print("------------------------------------------------------")
        print("mitosis_probability, before attack:", "%.10f" % float(results[6]))
        print("mitosis_probability, after attack:", "%.10f" % results[3])
        self.prediction_change.append(float(results[6]) - float(results[3]))
        self.print_single_attack_performance(results[4], results[5])
        print("------------------------------------------------------")
        self.store_single_attack_performance(results[4], results[5])
        self.save_attacked_image(results[0], image_path)

    def normal_perform_attack_and_get_result(self, img_data, image_path):
        few_pixel_attacker = FewPixelAttacker(
            dimensions=(64, 64), verbose=self.verbose_logging
        )
        print("----------[%s]----------" % Path(image_path).name)
        results = few_pixel_attacker.attack(
            img_data,
            pixel_count=5,
            maxiter=self.max_number_of_generations,
            popsize=self.population_size,
            threshold_fn=self.normal_threshold_fn,
            threshold_val=self.normal_threshold_val,
            minimize=self.min_fn_for_normal,
            color_bounds=self.near_black_bounds,
        )
        print("------------------------------------------------------")
        print("normal_probability, before attack:", "%.10f" % float(results[6]))
        print("normal_probability, after attack:", "%.10f" % results[3])
        self.prediction_change.append(float(results[3]) - float(results[6]))
        self.print_single_attack_performance(results[4], results[5])
        print("------------------------------------------------------")
        self.store_single_attack_performance(results[4], results[5])
        self.save_attacked_image(results[0], image_path)

    def get_image_name_and_extension(self, image_path):
        image_name_only = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name_only)
        return (name, ext)

    def generate_adversarial_patch_for_dataset(self, images_path, count: int = 10):
        if self.verbose_logging:
            print(images_path)
        image_paths = self.fshelper.read_images_from_folder(images_path, count)

        for image_path in image_paths:
            # convert image to numpy array
            data = np.array(Image.open(image_path))
            (name, ext) = self.get_image_name_and_extension(image_path)
            if "mitosis_" in name:
                self.patch_type = "mitosis"
                self.mitosis_perform_attack_and_get_result(data, image_path)
            else:
                self.patch_type = "normal"
                self.normal_perform_attack_and_get_result(data, image_path)

    def generate_adversarial_image_patches_for_mitosis_imageset(self, count: int = 10):
        self.patch_type = "mitosis"
        if self.verbose_logging:
            print(self.mitosis_images_path)
        image_paths = self.fshelper.read_images_from_folder(
            self.mitosis_images_path, count
        )

        for img_path in image_paths:
            # convert image to numpy array
            data = np.array(Image.open(img_path))
            self.mitosis_perform_attack_and_get_result(data, img_path)

    def generate_adversarial_image_patches_for_normal_imageset(self, count: int = 10):
        self.patch_type = "normal"
        if self.verbose_logging:
            print(self.normal_images_path)
        image_paths = self.fshelper.read_images_from_folder(
            self.normal_images_path, count
        )

        for img_path in image_paths:
            # convert image to numpy array
            data = np.array(Image.open(img_path))
            self.normal_perform_attack_and_get_result(data, img_path)
