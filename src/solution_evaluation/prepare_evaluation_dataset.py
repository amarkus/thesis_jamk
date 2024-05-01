from pathlib import Path
from random import choices
from src.lib.filesystemhelper import FileSystemHelper
import os
import shutil

# Get current working directory path
fshelper = FileSystemHelper()
working_directory = fshelper.current_working_directory()
basepath = os.path.normpath(os.path.join(working_directory, "src/dataset"))
file_type = ".png"
source_directory_originals = fshelper.get_dir_path(basepath, "original_patches")
source_directory_adversarials = fshelper.get_dir_path(basepath, "adversarial_patches")
target_directory = fshelper.get_dir_path(basepath, "evaluation_patches")
fshelper.create_directory(target_directory)


def copy_dataset_to_target_dir(
    source_dir, destination_dir, sample_count, file_type=".png"
):
    list_of_files = list(Path(source_dir).rglob("*" + file_type))

    for file_to_copy in list_of_files[:sample_count]:
        image_name = os.path.basename(file_to_copy)
        target_image_path = os.path.normpath(os.path.join(destination_dir, image_name))

        print("copying: {} to :{}".format(file_to_copy, target_image_path))
        shutil.copy(file_to_copy, target_image_path)


# Copy original images to demonstration/evaluation directory
copy_dataset_to_target_dir(
    source_dir=source_directory_originals,
    destination_dir=target_directory,
    sample_count=100,
)

# Copy adversarial images to demonstration/evaluation directory
copy_dataset_to_target_dir(
    source_dir=source_directory_adversarials,
    destination_dir=target_directory,
    sample_count=100,
)
