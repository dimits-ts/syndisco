"""
Module responsible for exporting discussions and their annotations in CSV
format.
"""

# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2025 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

import os
import json
import hashlib
from pathlib import Path
from typing import Iterable

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
    df = df.rename(columns={"id": "conv_id"})

    # Filter out non-persona information
    # assumes context and instructions are shared across participants
    df.user_prompts = df.user_prompts.apply(
        lambda user_prompts: [
            json.loads(user_prompt)["persona"] for user_prompt in user_prompts
        ]
    )

    # Select only current user prompt
    df["user_prompt"] = _select_user_prompt(df)

    # Merge moderator and user prompts
    df["is_moderator"] = _is_moderator(df.moderator, df.user)
    df.user_prompt = df.moderator_prompt.where(df.is_moderator, df.user_prompt)

    df["message_id"] = _generate_message_hash(df.conv_id, df.message)
    df["message_order"] = _add_message_order(df)

    traits_df = pd.concat(
        list(df.user_prompt.apply(_process_traits))
    ).reset_index(drop=True)
    del traits_df["username"]

    # Remove unused columns
    del df["user_prompts"]
    del df["user_prompt"]
    del df["users"]
    del df["moderator"]
    del df["moderator_prompt"]

    full_df = pd.concat([df, traits_df], axis=1)
    return full_df


def import_annotations(annot_dir: str | Path) -> pd.DataFrame:
    """
    Import annotation data from JSON files in a directory and process it
    into a DataFrame.

    This function reads JSON files containing annotation data, processes the
    data to standardize columns, and includes structured user traits.

    :param annot_dir: Directory containing JSON files with annotation data.
    :type annot_dir: str | Path
    :return: A DataFrame containing processed annotation data.
    :rtype: pd.DataFrame
    """
    annot_dir = Path(annot_dir)
    df = _read_annotations(annot_dir)
    df = df.reset_index(drop=True)
    df = _rename_annot_df_columns(df)

    # Generate unique message ID and message order
    df["message_id"] = _generate_message_hash(df.conv_id, df.message)
    df["message_order"] = _add_message_order(df)
    df = _group_all_but_one(df, "annot_personality_characteristics")
    return df


def _read_annotations(annot_dir: Path) -> pd.DataFrame:
    """
    Read annotation data from JSON files and convert it into a DataFrame.

    This function recursively reads all JSON files in the specified directory,
    extracts annotation data in raw form, and formats it into a DataFrame.

    :param annot_dir: Directory containing JSON files with annotation data.
    :type annot_dir: Path
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


def _rename_annot_df_columns(df):
    # Identify persona columns
    persona_prefix = "annotator_prompt.persona."
    rename_map = {
        col: "annot_" + col.replace(persona_prefix, "")
        for col in df.columns
        if col.startswith(persona_prefix)
    }
    # Apply renaming
    return df.rename(columns=rename_map)


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


def _select_user_prompt(df):
    selected_user_prompts = []

    for _, row in df.iterrows():
        curr_username = row["user"]
        if curr_username == row["moderator"]:
            # Use moderator prompt
            selected_user_prompts.append(
                json.loads(row["moderator_prompt"])["persona"]
            )
            continue

        user_prompts = row["user_prompts"]
        matched_prompt = next(
            (p for p in user_prompts if p["username"] == curr_username),
            None,
        )
        if matched_prompt is None:
            raise ValueError(
                f"No matching prompt found for username: {curr_username}"
            )

        selected_user_prompts.append(matched_prompt)

    return selected_user_prompts


def _process_traits(user_prompt: dict) -> pd.DataFrame:
    """
    Process traits extracted from messages into a structured DataFrame.

    :param series: Series containing traits in dictionary format.
    :type series: pd.Series
    :return: A DataFrame with extracted traits as columns.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame([user_prompt])


def _group_all_but_one(df: pd.DataFrame, to_list_col: str) -> pd.DataFrame:
    grouping_columns = [col for col in df.columns if col != to_list_col]
    aggregated_df = (
        df.groupby(grouping_columns, as_index=False)
        .agg({to_list_col: list})
        .reset_index(drop=True)
    )
    return aggregated_df


def _generate_message_hash(
    conv_ids: Iterable[str], messages: Iterable[str]
) -> list[str]:
    ls = []
    for conv_id, message in zip(conv_ids, messages):
        hashed_message = hashlib.md5(
            f"{conv_id}_{message}".encode()
        ).hexdigest()
        ls.append(hashed_message)
    return ls


def _add_message_order(df: pd.DataFrame) -> pd.Series:
    i = 1
    last_conv_id = -1
    last_message_id = -1
    numbers = []

    for _, row in df.iterrows():
        new_conv_id = row["conv_id"]
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
