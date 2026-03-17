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

    df["user_prompts"] = df["user_prompts"].apply(
        lambda prompts: [
            {
                "persona": json.loads(p)["persona"],
                "instructions": json.loads(p)["instructions"],
            }
            for p in prompts
        ]
    )

    # Select persona per message (user or moderator)
    df["persona"], df["prompt"] = _select_persona_and_prompt(df)

    # Moderator flag
    df["is_moderator"] = _is_moderator(df["moderator"], df["user"])

    # Message-level identifiers
    df["message_id"] = _generate_message_hash(df.conv_id, df.message)
    df["message_order"] = _add_message_order(df)

    # Drop unused columns
    df = df.drop(
        columns=["user_prompts", "users", "moderator_prompt", "moderator"]
    )

    return df


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

    # Generate unique message ID and message order
    df["message_id"] = _generate_message_hash(df.conv_id, df.message)
    df = _group_all_but_one(df)
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
    rows = []

    for file_path in annot_dir.rglob("*.json"):
        with open(file_path, "r", encoding="utf8") as fin:
            conv = json.load(fin)

        conv = pd.json_normalize(conv)
        conv = conv.explode("logs")

        conv["message"] = conv.logs.apply(lambda x: x[0])
        conv["annotation"] = conv.logs.apply(lambda x: x[1])

        conv = conv.drop(columns=["logs"])
        rows.append(conv)

    return pd.concat(rows, ignore_index=True)


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
        raise ValueError(f"{conv_dir} is not a directory or does not exist")

    rows = []
    for file_path in conv_dir.rglob("*.json"):
        with open(file_path, "r", encoding="utf8") as fin:
            conv = json.load(fin)

        base = pd.json_normalize(conv)
        logs = base.explode("logs")

        logs["user"] = logs["logs"].apply(lambda x: x["name"])
        logs["message"] = logs["logs"].apply(lambda x: x["text"])
        logs["model"] = logs["logs"].apply(lambda x: x["model"])

        logs = logs.drop(columns=["logs"])
        rows.append(logs)

    return pd.concat(rows, ignore_index=True)


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


def _select_persona_and_prompt(
    df: pd.DataFrame,
) -> tuple[list[dict], list[str]]:
    personas = []
    prompts = []

    for _, row in df.iterrows():
        username = row["user"]

        # Moderator message
        if username == row["moderator"]:
            moderator_prompt = json.loads(row["moderator_prompt"])
            personas.append(moderator_prompt["persona"])
            prompts.append(moderator_prompt["instructions"])
            continue

        # Regular user
        match = next(
            (
                p
                for p in row["user_prompts"]
                if p["persona"]["username"] == username
            ),
            None,
        )

        if match is None:
            raise ValueError(
                f"No matching persona found for username: {username}"
            )

        personas.append(match["persona"])
        prompts.append(match["instructions"])

    return personas, prompts


def _group_all_but_one(df: pd.DataFrame) -> pd.DataFrame:
    grouping_columns = [
        c for c in df.columns if c not in ["annotation", "annotator_prompt"]
    ]

    return df.groupby(grouping_columns, as_index=False).agg(
        {"annotation": list, "annotator_prompt": list}
    )


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
