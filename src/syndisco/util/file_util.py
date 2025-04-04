"""
SynDisco: Automated experiment creation and execution using only LLM agents
Copyright (C) 2025 Dimitris Tsirmpas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at tsirbasdim@gmail.com
"""

import os
import json
import datetime
import logging

from typing import Any, Optional
from pathlib import Path


logger = logging.getLogger(Path(__file__).name)


def read_files_from_directory(directory: str | Path) -> list[str]:
    """Reads all files from a given directory.

    :param directory: the root directory from which to load files
        (NOT recursively!)
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
    """Return the contents of a file

    :param path: the path of the file
    :type path: str | Path
    :return: the file's contents
    :rtype: str
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


def dict_to_json(dictionary: dict[str, Any], output_path: str | Path) -> None:
    """
    Write a python dictionary to a file in JSON format.

    :param dictionary: The dictionary to be serialized
    :type dictionary: dict[str, Any]
    :param output_path: The output path to the file. If the parent directories
        do not exist, they will be created.
    :type output_path: str | Path
    """
    ensure_parent_directories_exist(output_path)

    with open(output_path, "w", encoding="utf8") as fout:
        json.dump(dictionary, fout, indent=4)


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
) -> Path:
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
    datetime_name = (
        datetime.datetime.now().strftime(timestamp_format) + file_ending
    )
    path = ""
    if output_dir is None:
        path = Path(datetime_name)
    else:
        path = Path(os.path.join(output_dir, datetime_name))

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    return path
