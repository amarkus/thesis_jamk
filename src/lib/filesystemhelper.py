import glob
import re
import os
import numpy as np
import shutil
from pathlib import Path
from random import choices
from PIL import Image


class FileSystemHelper:

    def __init__(self):
        # Common properties
        self.working_directory = self.current_working_directory()

    def create_directory(self, path):
        folderExist = os.path.exists(path)
        if not folderExist:
            os.makedirs(path)
            print(f"Created folder {path}")

    # If you want to change your working directory, use this method
    def current_working_directory(self, path=""):
        cwd = os.getcwd() if path == "" else os.chdir(path)
        self.working_directory = cwd
        print(f"Current working directory is: {cwd}")
        return cwd

    def get_dir_path(
        self, current_working_dir, relative_path, verbose_logging: bool = False
    ):
        path = os.path.normpath(os.path.join(current_working_dir, relative_path))
        if verbose_logging:
            print("path: %s" % path)
        return path

    def copy_random_set_for_attack_targets(
        self, source_dir, destination_dir, sample_count, file_type, prefix
    ):
        list_of_files = list(Path(source_dir).rglob("*" + file_type))
        test_set = choices(list_of_files, k=sample_count)
        count = 1
        for file_to_copy in test_set:
            # Initial new name
            base, extension = os.path.splitext(file_to_copy)
            new_name = f"{prefix}_{count}{extension}"
            new_file_path = os.path.join(destination_dir, new_name)
            print("copying: {} to :{}".format(file_to_copy, new_file_path))
            shutil.copy(file_to_copy, new_file_path)
            count += 1

    def read_images_from_folder(
        self,
        filepath: str,
        count: int = 10,
        sort_by_number: bool = True,
        verbose_logging: bool = False,
        recursive_subfolders: bool = False,
    ):
        image_list = []
        search_path = "%s\*.png" % filepath
        if verbose_logging:
            print("search_path: %s" % search_path)
        images = glob.glob(search_path)
        if recursive_subfolders:
            search_path = f"{filepath}/**/*.png"
            images = glob.glob(search_path, recursive=True)
        if verbose_logging:
            print("imagecount: %s" % images[:count])
        for image in images[:count]:
            image_list.append(image)
        if sort_by_number:
            image_list.sort(key=lambda f: int(re.sub("\D", "", f)))
        return image_list

    def read_image_as_array(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = np.array(img.resize((64, 64), Image.Resampling.LANCZOS))
        img = img / 255.0
        img = img[np.newaxis, :, :, :]

        return img
