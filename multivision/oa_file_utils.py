import os
import random


def search_substring_from_folder(path, substr):
    files = os.listdir(path)
    for file in files:
        if substr in file:
            return os.path.join(path, file)
    return None

def list_all_files(dir_path):
    files = os.listdir(dir_path)
    return  [os.path.join(dir_path, name) for name in files]

def path_to_random_file(dir_path):
    dir_paths = list_all_files(dir_path)
    return random.choice(dir_paths)

