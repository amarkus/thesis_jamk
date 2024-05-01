import unittest
import os
import glob
import shutil
from src.preprocess_dataset.dataset import MitosisDataset

# Passing the -v option to your test script will instruct
# unittest.main() to enable a higher level of verbosity


class TestDatasetSplitting(unittest.TestCase):
    def setUp(self):

        self.mitosis_dataset = MitosisDataset(
            image_size=(64, 64), batch_size=64, verbose=False
        )
        directory = os.path.dirname(__file__)
        self.patches_folder = os.path.normpath(os.path.join(directory, "patches"))
        self.output_folder_fixed = os.path.normpath(
            os.path.join(directory, "test_output_fixed")
        )
        self.output_folder_ratio = os.path.normpath(
            os.path.join(directory, "test_output_ratio")
        )

        # Make test output_folder_fixed directory
        if os.path.isdir(self.output_folder_fixed) == False:
            os.mkdir(self.output_folder_fixed)
        # Make test output_folder_ratio directory
        if os.path.isdir(self.output_folder_ratio) == False:
            os.mkdir(self.output_folder_ratio)

    def test_split_dataset_fixed(self):
        # Run dataset split to create train/val/test hierarchy
        self.mitosis_dataset.split_dataset_fixed(
            input_folder=self.patches_folder,
            output_folder=self.output_folder_fixed,
            seed_value=1,
            items_per_set=(2, 1, 1),
            oversampling=False,
            move_instead_of_copy=False,
        )
        # Count resulting folders
        folder = glob.glob(f"{self.output_folder_fixed}/*")
        number_of_subfolders = len(folder)
        # Did we get the train/val/test folders?
        self.assertEqual(number_of_subfolders, 3)

    def test_split_dataset_ratio(self):
        # Run dataset split to create train/val/test hierarchy
        training_data = self.mitosis_dataset.split_dataset_ratio(
            input_folder=self.patches_folder,
            output_folder=self.output_folder_ratio,
            seed_value=1,
            split_ratio=(0.7, 0.2, 0.1),
            move_instead_of_copy=False,
        )
        # Count resulting folders
        folder = glob.glob(f"{self.output_folder_ratio}/*")
        number_of_subfolders = len(folder)
        # Did we get the train/val/test folders?
        self.assertEqual(number_of_subfolders, 3)

    def tearDown(self):
        # remove old test output_folder_fixed directories
        if os.path.isdir(self.output_folder_fixed) == True:
            shutil.rmtree(self.output_folder_fixed)
        # remove old test output_folder_ratio directories
        if os.path.isdir(self.output_folder_ratio) == True:
            shutil.rmtree(self.output_folder_ratio)


if __name__ == "__main__":
    print("Running test set for data splitting in class: MitosisDataset")
    unittest.main(verbosity=2)
