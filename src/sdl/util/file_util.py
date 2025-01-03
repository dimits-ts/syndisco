import os
import json
import datetime

from typing import Any, Optional
from pathlib import Path


def read_files_from_directory(directory: str | Path) -> list[str]:
    """Reads all files from a given directory.

    :param directory: the root directory from which to load files (NOT recursively!)
    :type directory: str | Path

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


def read_file(path: str | Path) -> str:
    """Read a plain text or JSON file depending on its extension

    :param path: the path of the file
    :type path: str | Path
    :return: the file's contents
    :rtype: str | dict[str, Any]
    """
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON file
    :param path: the path of the file
    :type path: str | Path
    :return: the file's contents
    :rtype: dict[str, Any]
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_parent_directories_exist(output_path: str | Path) -> None:
    """
    Create all parent directories if they do not exist.
    :param output_path: the path for which parent dirs will be generated
    :type output_path: str | Path
    """
    # Extract the directory path from the given output path
    directory = os.path.dirname(output_path)

    # Create all parent directories if they do not exist
    if directory:
        os.makedirs(directory, exist_ok=True)


def generate_datetime_filename(
    output_dir: Optional[str | Path] = None,
    timestamp_format: str = "%y-%m-%d-%H-%M",
    file_ending: str = "",
) -> str:
    """
    Generate a filename based on the current date and time.

    :param output_dir: The path to the generated file, defaults to None
    :type output_dir: str | Path, optional
    :param timestamp_format: strftime format, defaults to "%y-%m-%d-%H-%M"
    :type timestamp_format: str, optional
    :param file_ending: The ending of the file (e.g '.json')
    :type file_ending: str
    :return: the full path for the generated file
    :rtype: str
    """
    datetime_name = datetime.datetime.now().strftime(timestamp_format) + file_ending

    if output_dir is None:
        return datetime_name
    else:
        return os.path.join(output_dir, datetime_name)
