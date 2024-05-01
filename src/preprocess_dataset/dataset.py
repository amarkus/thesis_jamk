import splitfolders
from keras.preprocessing.image import ImageDataGenerator
from src.lib.filesystemhelper import FileSystemHelper


class MitosisDataset:

    def __init__(self, image_size=(64, 64), batch_size=64, verbose=False):
        self.image_size = image_size
        self.batch_size = batch_size
        fshelper = FileSystemHelper()
        self.working_directory = fshelper.current_working_directory()

    def load_training_data(
        self, directory="./dataset", training_data_folder="train", augment=True
    ):
        train_datagen = ImageDataGenerator(rescale=1.0 / 255)
        train_datagen_augmented = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        train_datagen = train_datagen if not augment else train_datagen_augmented

        train_generator = train_datagen.flow_from_directory(
            directory,
            classes=[training_data_folder],
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="input",
            subset="training",
            shuffle=True,
            seed=42,
        )

        return train_generator

    def load_validation_data(self, directory="./dataset", validation_data_folder="val"):
        validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
        validation_generator = validation_datagen.flow_from_directory(
            directory,
            classes=[validation_data_folder],
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="input",
            subset=None,
            shuffle=False,
            seed=42,
        )

        return validation_generator

    def load_test_data(self, directory="./dataset", test_data_folder="test"):
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        test_generator = test_datagen.flow_from_directory(
            directory,
            classes=[test_data_folder],
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="input",
            subset=None,
            shuffle=False,
            seed=42,
        )

        return test_generator

    def split_dataset_fixed(
        self,
        input_folder="./patches",
        output_folder="./dataset",
        seed_value=1337,
        items_per_set=(7000, 2000, 1000),
        oversampling=False,
        move_instead_of_copy=False,
    ):
        # Split train/val/test with a fixed number of items,
        # e.g. `(7000, 2000, 1000)`.
        splitfolders.fixed(
            input_folder,
            output=output_folder,
            seed=seed_value,
            fixed=items_per_set,
            oversample=oversampling,
            group_prefix=None,
            move=move_instead_of_copy,
        )

    def split_dataset_ratio(
        self,
        input_folder,
        output_folder,
        seed_value=1337,
        split_ratio=(0.7, 0.2, 0.1),
        move_instead_of_copy=False,
    ):
        #   The default ratio used with this data set split:
        #       train(train): 0.7
        #       validation(val): 0.2
        #       testing(test): 0.1
        splitfolders.ratio(
            input_folder,
            output=output_folder,
            seed=seed_value,
            ratio=split_ratio,
            group_prefix=None,
            move=move_instead_of_copy,
        )
