import pandas as pd

import os
import json
import re
from pathlib import Path


def import_conversations(conv_dir: str | Path) -> pd.DataFrame:
    df = _read_conversations(conv_dir)
    df = df.reset_index(drop=True)

    # remove useless columns
    del df["users"]

    # from having a list of all user_prompts -> having only the relevant prompt
    selected_prompt = _select_user_prompt(df)
    df["user_prompt"] = selected_prompt
    del df["user_prompts"]

    # merge the moderator_prompt and user_prompt columns
    df["is_moderator"] = _is_moderator(df.moderator, df.user)
    df.user_prompt = df.moderator_prompt.where(df.is_moderator, df.user_prompt)
    del df["moderator"], df["moderator_prompt"]

    # add attributes for each user as rows
    df2 = process_traits(df.user_prompt.apply(_extract_traits)).reset_index()
    del df2["username"]
    df = pd.concat([df, df2], axis=1)
    return df


def import_annotations(annot_dir: str | Path, include_sdb_info: bool=False) -> pd.DataFrame:
    """
    Import annotation data from a directory containing JSON files and convert them to a DataFrame.

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
    annot_df = _read_annotations(annot_dir=annot_dir)
    annot_df = annot_df.reset_index(drop=True)

    if include_sdb_info:
        # add attributes for each user as rows
        traits_df = process_traits(annot_df.annotator_prompt.apply(_extract_traits)).reset_index()
        annot_df = pd.concat([annot_df, traits_df], axis=1)
        del annot_df["special_instructions"]
    
    return annot_df


def _read_annotations(annot_dir: str | Path) -> pd.DataFrame:
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
    file_paths = _list_files_recursive(annot_dir)
    rows = []

    for file_path in file_paths:
        with open(file_path, "r") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv.explode("logs")
        # get name, not path of parent directory
        conv["annotation_variant"] = os.path.basename(os.path.dirname(file_path))
        conv["message"] = conv.logs.apply(lambda x: x[0])
        conv["annotation"] = conv.logs.apply(lambda x: x[1])

        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    return full_df


def _read_conversations(conv_dir: str | Path) -> pd.DataFrame:
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


def _is_moderator(moderator_name: pd.Series, username: pd.Series) -> pd.Series:
    return moderator_name == username


# code adapted from https://www.geeksforgeeks.org/python-list-all-files-in-directory-and-subdirectories/
def _list_files_recursive(start_path: str | Path) -> list[str]:
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


def process_traits(series):
    """
    Processes a pandas Series of strings containing schema-like traits
    and converts them into a DataFrame with a column for each attribute.

    Parameters:
        series (pd.Series): The input pandas Series containing trait schemas.

    Returns:
        pd.DataFrame: A DataFrame where each column represents an attribute.
    """
    traits_list = series
    return pd.DataFrame(traits_list.tolist())


def _extract_traits(message):
    """
    Extracts attribute-value pairs from the 'traits' section of the message.

    Parameters:
        message (str): The input message containing traits.

    Returns:
        dict: A dictionary of extracted traits as attribute-value pairs.
    """
    if message is None:
        return {}

    # Extract the traits section
    traits_match = re.search(
        r"Your traits: (.+?) Your instructions:", message, re.DOTALL
    )
    if not traits_match:
        return {}

    traits_section = traits_match.group(1).strip()

    # Split traits into individual attribute-value pairs
    traits = {}
    for match in re.finditer(r'(\w+):\s*(\[.*?\]|".*?"|\'.*?\'|\S+)', traits_section):
        key = match.group(1)
        value = match.group(2)

        # Convert list-like and quoted values to appropriate Python objects
        try:
            if value.startswith("[") and value.endswith("]"):
                value = eval(value)  # Safely parse list-like values
            elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value.strip("'\"")
            else:
                value = value.replace(",", "")
        except Exception:
            pass  # Leave the value as a string if parsing fails

        traits[key] = value

    return traits
