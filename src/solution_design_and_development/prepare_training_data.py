import os
from src.lib.filesystemhelper import FileSystemHelper
from src.preprocess_dataset.dataset import MitosisDataset

# Get current working directory path
fshelper = FileSystemHelper()
working_directory = fshelper.current_working_directory()
mitosis_dataset = MitosisDataset(image_size=(64, 64), batch_size=64, verbose=False)

# Set input & output directories
patches_folder = os.path.normpath(
    os.path.join(working_directory, "src/dataset/patches")
)
output_folder_fixed = os.path.normpath(
    os.path.join(working_directory, "src/dataset/training_data")
)

# Make test output_folder_fixed directory
if os.path.isdir(output_folder_fixed) == False:
    os.mkdir(output_folder_fixed)

# Run dataset split to create train/val/test hierarchy
mitosis_dataset.split_dataset_fixed(
    input_folder=patches_folder,
    output_folder=output_folder_fixed,
    seed_value=1,
    items_per_set=(7, 2, 1),
    # items_per_set=(7000, 2000, 1000),
    oversampling=False,
    move_instead_of_copy=False,
)
