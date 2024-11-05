import os
import json
from typing import Any


def read_files_from_directory(directory: str) -> list[str]:
    """Reads all files from a given directory.

    :param directory: the root directory from which to load files (NOT recursively!)
    :type directory: str

    :raises ValueError: if the directory does not exist
    :return: Returns a list of parsed file content.
    :rtype: list[str | dict]
    """
    files_list = []

    # Check if directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist.")

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        data = read_file(file_path)
        files_list.append(data)

    return files_list


def read_file(path: str) -> str:
    """Read a plain text or JSON file depending on its extension

    :param path: the path of the file
    :type path: str
    :return: the file's contents
    :rtype: str | dict[str, Any]
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_json_file(path: str):
    """Read a JSON file
    :param path: the path of the file
    :type path: str
    :return: the file's contents
    :rtype: dict[str, Any]
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
