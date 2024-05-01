import unittest
import os
from src.preprocess_dataset.dataset import MitosisDataset

# Passing the -v option to your test script will instruct
# unittest.main() to enable a higher level of verbosity


class TestDatasetLoading(unittest.TestCase):
    def setUp(self):

        self.mitosis_dataset = MitosisDataset(
            image_size=(64, 64), batch_size=64, verbose=False
        )
        directory = os.path.dirname(__file__)
        self.test_dataset_folder = os.path.normpath(
            os.path.join(directory, "test_dataset")
        )

    def test_load_training_data(self):
        training_data = self.mitosis_dataset.load_training_data(
            directory=self.test_dataset_folder
        )
        self.assertEqual(training_data.samples, 10)

    def test_load_validation_data(self):
        validation_data = self.mitosis_dataset.load_validation_data(
            directory=self.test_dataset_folder
        )
        self.assertEqual(validation_data.samples, 10)

    def test_load_test_data(self):
        test_data = self.mitosis_dataset.load_test_data(
            directory=self.test_dataset_folder
        )
        self.assertEqual(test_data.samples, 10)


if __name__ == "__main__":
    print("Running test set for data loading in class: MitosisDataset")
    unittest.main(verbosity=2)
