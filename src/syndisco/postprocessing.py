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
import re
import ast
from pathlib import Path

import pandas as pd


def import_discussions(conv_dir: Path) -> pd.DataFrame:
    """
    Import discussion output (logs) from JSON files in a directory and process
     it into a DataFrame.

    This function reads JSON files containing conversation data, processes the
     data to
    standardize columns, and adds derived attributes such as user traits and
     prompts.

    :param conv_dir: Directory containing JSON files with conversation data.
    :type conv_dir: str | Path
    :return: A DataFrame containing processed conversation data.
    :rtype: pd.DataFrame
    """
    df = _read_conversations(conv_dir)
    df = df.reset_index(drop=True)

    # Remove unused columns
    del df["users"]

    # Select relevant user prompts
    selected_prompt = _select_user_prompt(df)
    df["user_prompt"] = selected_prompt
    del df["user_prompts"]

    # Merge moderator and user prompts
    df["is_moderator"] = _is_moderator(df.moderator, df.user)
    df.user_prompt = df.moderator_prompt.where(df.is_moderator, df.user_prompt)
    del df["moderator"], df["moderator_prompt"]

    df["message_id"] = df.apply(
        lambda row: _generate_message_hash(row["id"], row["message"]), axis=1
    )
    df["message_order"] = _add_message_order(df)

    # Extract user traits and add them as attributes
    df2 = _process_traits(df.user_prompt.apply(_extract_traits)).reset_index()
    del df2["username"]
    df = pd.concat([df, df2], axis=1)
    return df


def import_annotations(annot_dir: str | Path) -> pd.DataFrame:
    """
    Import annotation data from JSON files in a directory and process it
    into a DataFrame.

    This function reads JSON files containing annotation data, processes the
    data to standardize columns, and optionally includes SDB information for
    annotators.

    :param annot_dir: Directory containing JSON files with annotation data.
    :type annot_dir: str | Path
    :return: A DataFrame containing processed annotation data.
    :rtype: pd.DataFrame
    """
    annot_df = _read_annotations(annot_dir)
    annot_df = annot_df.reset_index(drop=True)

    # Add annotator traits
    traits_df = _process_traits(
        annot_df.annotator_prompt.apply(_extract_traits)
    ).reset_index()
    annot_df = pd.concat([annot_df, traits_df], axis=1)
    del annot_df["special_instructions"]

    return annot_df


def _read_annotations(annot_dir: str | Path) -> pd.DataFrame:
    """
    Read annotation data from JSON files and convert it into a DataFrame.

    This function recursively reads all JSON files in the specified directory,
    extracts annotation data in raw form, and formats it into a DataFrame.

    :param annot_dir: Directory containing JSON files with annotation data.
    :type annot_dir: str | Path
    :return: A DataFrame containing raw annotation data.
    :rtype: pd.DataFrame
    """
    file_paths = _list_files_recursive(annot_dir)
    rows = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf8") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv.explode("logs")
        conv["annotation_variant"] = os.path.basename(
            os.path.dirname(file_path)
        )
        conv["message"] = conv.logs.apply(lambda x: x[0])
        conv["annotation"] = conv.logs.apply(lambda x: x[1])

        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    return full_df


def _read_conversations(conv_dir: Path) -> pd.DataFrame:
    """
    Read conversation data from JSON files and convert it into a DataFrame.

    This function recursively reads all JSON files in the specified directory,
    extracts conversation data in raw form, and formats it into a DataFrame.

    :param conv_dir: Directory containing JSON files with conversation data.
    :type conv_dir: str | Path
    :return: A DataFrame containing raw conversation data.
    :rtype: pd.DataFrame
    """
    if not conv_dir.is_dir():
        raise ValueError(
            f"{conv_dir} is not a directory or does not exist"
        ) from None

    file_paths = _list_files_recursive(conv_dir)

    if len(file_paths) == 0:
        raise ValueError(
            "No discussions found in directory ", conv_dir
        ) from None
    rows = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf8") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv.explode("logs")
        conv["conv_variant"] = os.path.basename(os.path.dirname(file_path))
        conv["user"] = conv.logs.apply(lambda x: x["name"])
        conv["message"] = conv.logs.apply(lambda x: x["text"])
        conv["model"] = conv.logs.apply(lambda x: x["model"])
        del conv["logs"]
        rows.append(conv)

    full_df = pd.concat(rows)
    return full_df


def _is_moderator(moderator_name: pd.Series, username: pd.Series) -> pd.Series:
    """
    Determine if a user is the moderator.

    :param moderator_name: Series of moderator names.
    :type moderator_name: pd.Series
    :param username: Series of usernames.
    :type username: pd.Series
    :return: A Series indicating whether each user is the moderator.
    :rtype: pd.Series
    """
    return moderator_name == username


def _list_files_recursive(start_path: str | Path) -> list[str]:
    """
    Recursively list all files in a directory and its subdirectories.

    :param start_path: The starting directory path.
    :type start_path: str | Path
    :return: A list of file paths.
    :rtype: list[str]
    """
    all_files = []
    for root, _, files in os.walk(start_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def _select_user_prompt(df: pd.DataFrame) -> list[str]:
    """
    Select relevant user prompts for each conversation entry.

    :param df: DataFrame containing user prompts and usernames.
    :type df: pd.DataFrame
    :return: A list of selected user prompts.
    :rtype: list[str]
    """
    selected_user_prompts = []
    for row in df.itertuples():
        prompt = _extract_user_prompt(row.user_prompts, row.user)
        selected_user_prompts.append(prompt)
    return selected_user_prompts


def _extract_user_prompt(
    user_prompts: list[str], username: str | None
) -> str | None:
    """
    Extract the prompt associated with a specific username.

    :param user_prompts: List of user prompts.
    :type user_prompts: list[str]
    :param username: The username for which to extract the prompt.
    :type username: str | None
    :return: The relevant user prompt, or None if not found.
    :rtype: str | None
    """
    if username is None:
        return None

    for user_prompt in user_prompts:
        if username in user_prompt:
            return user_prompt
    return None


def _process_traits(series: pd.Series) -> pd.DataFrame:
    """
    Process traits extracted from messages into a structured DataFrame.

    :param series: Series containing traits in dictionary format.
    :type series: pd.Series
    :return: A DataFrame with extracted traits as columns.
    :rtype: pd.DataFrame
    """
    traits_list = series
    return pd.DataFrame(traits_list.tolist())


def _extract_traits(message: str | None) -> dict:
    """
    Extract traits from a message's 'traits' section.

    :param message: The input message containing traits.
    :type message: str | None
    :return: A dictionary of extracted traits.
    :rtype: dict
    """
    if message is None:
        return {}

    traits_match = re.search(
        r"Your traits: (.+?) Your instructions:", message, re.DOTALL
    )
    if not traits_match:
        return {}

    traits_section = traits_match.group(1).strip()

    traits = {}
    for match in re.finditer(
        r'(\w+):\s*(".*?"|\[.*?\]|[\w\s]+)(?=,|$)', traits_section
    ):
        key = match.group(1)
        value = match.group(2).strip()

        try:
            if value.startswith("[") and value.endswith("]"):
                value = ast.literal_eval(value)
            elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value.strip("'\"")
        except ValueError:
            pass

        traits[key] = value

    return traits


def _generate_message_hash(conv_id: str, message: str, hash_func=hash) -> str:
    return hash_func(hash_func(conv_id) + hash_func(message))


def _add_message_order(df: pd.DataFrame) -> pd.Series:
    i = 1
    last_conv_id = -1
    last_message_id = -1
    numbers = []

    for _, row in df.iterrows():
        new_conv_id = row["id"]
        new_message_id = row["message_id"]

        if new_conv_id != last_conv_id:
            last_conv_id = new_conv_id
            last_message_id = new_message_id
            i = 1
        elif new_message_id != last_message_id:
            i += 1
            last_message_id = new_message_id

        numbers.append(i)
    return pd.Series(numbers)
