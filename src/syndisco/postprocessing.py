"""
Module responsible for exporting discussions and their annotations in CSV
format.
"""

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
from pathlib import Path
from typing import Iterable
from io import StringIO

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
    """
    Convert a list of JSON conversation files into a flat CSV format.
    
    Args:
        json_files (list[str or Path]): List of file paths to JSON conversation files.
        output_csv (str or Path): Output CSV path.
    """
    rows = []

    for file in conv_dir.rglob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        conv_id = data.get("id")
        timestamp_conv = data.get("timestamp")
        ctx_length_conv = data.get("ctx_length")
        conv_variant = None

        message_order = 1

        # Build persona + prompt lookup
        persona_by_user = {}
        for entry in data.get("user_prompts", []):
            persona = entry.get("persona", {})
            username = persona.get("username")
            if username:
                persona_by_user[username] = {
                    "user_prompt_context": entry.get("context"),
                    "user_prompt_instructions": entry.get("instructions"),
                    "user_prompt_type": entry.get("type"),
                    "age": persona.get("age"),
                    "sex": persona.get("sex"),
                    "sexual_orientation": persona.get("sexual_orientation"),
                    "demographic_group": persona.get("demographic_group"),
                    "current_employment": persona.get("current_employment"),
                    "education_level": persona.get("education_level"),
                    "special_instructions": persona.get(
                        "special_instructions"
                    ),
                }

        # Add moderator
        moderator_prompt = data.get("moderator_prompt", {})
        moderator_persona = moderator_prompt.get("persona", {})
        moderator_username = moderator_persona.get("username", "moderator")

        persona_by_user[moderator_username] = {
            "user_prompt_context": moderator_prompt.get("context"),
            "user_prompt_instructions": moderator_prompt.get("instructions"),
            "user_prompt_type": moderator_prompt.get("type"),
            "age": moderator_persona.get("age"),
            "sex": moderator_persona.get("sex"),
            "sexual_orientation": moderator_persona.get("sexual_orientation"),
            "demographic_group": moderator_persona.get("demographic_group"),
            "current_employment": moderator_persona.get("current_employment"),
            "education_level": moderator_persona.get("education_level"),
            "special_instructions": moderator_persona.get(
                "special_instructions"
            ),
        }

        # Process logs
        for log in data.get("logs", []):
            user = log.get("name")
            message = log.get("text")
            model = log.get("model")

            persona_info = persona_by_user.get(user, {})

            rows.append(
                {
                    "conv_id": conv_id,
                    "timestamp_conv": timestamp_conv,
                    "ctx_length_conv": ctx_length_conv,
                    "conv_variant": conv_variant,
                    "user": user,
                    "message": message,
                    "model": model,
                    "is_moderator": (user == moderator_username),
                    "message_id": f"{conv_id}_{message_order}",
                    "message_order": message_order,
                    "user_prompt_context": persona_info.get(
                        "user_prompt_context"
                    ),
                    "user_prompt_instructions": persona_info.get(
                        "user_prompt_instructions"
                    ),
                    "user_prompt_type": persona_info.get("user_prompt_type"),
                    "age_conv": persona_info.get("age"),
                    "sex_conv": persona_info.get("sex"),
                    "sexual_orientation_conv": persona_info.get(
                        "sexual_orientation"
                    ),
                    "demographic_group_conv": persona_info.get(
                        "demographic_group"
                    ),
                    "current_employment_conv": persona_info.get(
                        "current_employment"
                    ),
                    "education_level_conv": persona_info.get(
                        "education_level"
                    ),
                    "special_instructions": persona_info.get(
                        "special_instructions"
                    ),
                }
            )

            message_order += 1

    df = pd.DataFrame(rows)
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


def _group_all_but_one(df: pd.DataFrame, to_list_col: str) -> pd.DataFrame:
    grouping_columns = [col for col in df.columns if col != to_list_col]
    aggregated_df = (
        df.groupby(grouping_columns, as_index=False)
        .agg({to_list_col: list})
        .reset_index(drop=True)
    )
    return aggregated_df


def _generate_message_hash(
    conv_ids: Iterable[str], messages: Iterable[str], hash_func=hash
) -> list[str]:
    ls = []
    for conv_id, message in zip(conv_ids, messages):
        ls.append(hash_func(hash_func(conv_id) + hash_func(message)))
    return ls
