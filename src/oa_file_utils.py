import os


def search_substring_from_folder(path, substr):
    files = os.listdir(path)
    for file in files:
        if substr in file:
            return os.path.join(path, file)
    return None
