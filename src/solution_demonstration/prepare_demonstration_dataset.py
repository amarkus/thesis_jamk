from src.lib.filesystemhelper import FileSystemHelper
import os

# Get current working directory path
fshelper = FileSystemHelper()
working_directory = fshelper.current_working_directory()
basepath = os.path.normpath(os.path.join(working_directory, "src/dataset"))
file_type = ".png"

# Use validation data as source for demonstration
source_dir_normal = fshelper.get_dir_path(basepath, "training_data/val/normal")
source_dir_mitosis = fshelper.get_dir_path(basepath, "training_data/val/mitosis")
target_dir_original = fshelper.get_dir_path(basepath, "demonstration_patches/original")

# Ensure target directories exist
fshelper.create_directory(target_dir_original)

# Copy originals to be used in demo (and for attack targets)
fshelper.copy_random_set_for_attack_targets(
    source_dir_normal, target_dir_original, 50, file_type, prefix="normal"
)

fshelper.copy_random_set_for_attack_targets(
    source_dir_mitosis, target_dir_original, 50, file_type, prefix="mitosis"
)
