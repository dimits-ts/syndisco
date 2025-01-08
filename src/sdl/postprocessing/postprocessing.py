import pandas as pd

import os
import json
import re


def import_conversations(conv_dir: str) -> pd.DataFrame:
    df = _read_conversations(conv_dir)
    df = _add_moderator_exists(df)
    
    selected_prompt = _select_user_prompt(df)
    df["user_prompt"] = selected_prompt
    del df["user_prompts"]

    return df


def _read_conversations(conv_dir: str) -> pd.DataFrame:
    """
    Import conversation data from a directory containing JSON files and convert them to a DataFrame.

    Recursively reads all JSON files from the specified directory,
    and extracts relevant fields. It also adds metadata about the conversation variant.

    :param conv_dir: Path to the root directory containing the conversation JSON files.
    :type conv_dir: str
    :return: A DataFrame with conversation data, including the ID, user prompts, messages,
             and conversation variant.
    :rtype: pd.DataFrame

    :example:
        >>> df = import_conversations("/path/to/conversation/data")
    """
    file_paths = _list_files_recursive(conv_dir)
    rows = []

    for file_path in file_paths:
        with open(file_path, "r") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv.explode("logs")
        # get name, not path of parent directory
        conv["conv_variant"] = os.path.basename(os.path.dirname(file_path))
        conv["user"] = conv.logs.apply(lambda x: x["name"])
        conv["message"] = conv.logs.apply(lambda x: x["text"])
        conv["model"] = conv.logs.apply(lambda x: x["model"])
        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    return full_df


# code adapted from https://www.geeksforgeeks.org/python-list-all-files-in-directory-and-subdirectories/
def _list_files_recursive(start_path: str) -> list[str]:
    """
    Recursively list all files in a directory and its subdirectories.

    :param start_path: The starting directory path. Defaults to the current directory.
    :type start_path: str, optional
    :return: A list of file paths.
    :rtype: list[str]

    :example:
       >>> file_paths = _files_from_dir_recursive("/path/to/data")
    """
    all_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def _add_moderator_exists(df: pd.DataFrame) -> pd.DataFrame:
    moderator_ids = set(df[df["user"] == "moderator"]["id"])
    df["moderator_exists"] = df["id"].apply(lambda x: x in moderator_ids)
    return df


def _select_user_prompt(df: pd.DataFrame) -> list[str]:
    selected_user_prompts = []
    for row in df.itertuples():
        prompt = _extract_user_prompt(row.user_prompts, row.user)
        selected_user_prompts.append(prompt)
    return selected_user_prompts


def _extract_user_prompt(user_prompts: list[str], username: str | None) -> str | None:
    if username is None:
        return None

    for user_prompt in user_prompts:
        if username in user_prompt:
            return user_prompt
    return None

