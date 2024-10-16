from sdl import conversation_io

import json
import uuid
import os
import argparse
import random
from typing import Any


CTX_PREFACE = ""
DEFAULT_MODERATOR_ATTRIBUTES = ["just", "strict", "understanding"]


def read_files_from_directory(directory: str, file_extension: str) -> list[str | dict]:
    """
    Reads all files with the specified extension from a given directory.
    Supports .json and .txt files.
    Returns a list of parsed file content.
    """
    files_list = []

    # Check if directory exists
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist.")

    # Loop through all files in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if file_path.endswith(file_extension):
            data = read_file(file_path, file_extension)
            files_list.append(data)

    return files_list


def read_file(path: str, file_extension: str) -> str | dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        if file_extension == ".json":
            return json.load(file)
        else:
            return file.read()


def generate_conv_config(
    personas: list[dict[str, Any]],
    topics: list[str],
    user_instructions: str,
    mod_instructions: str,
    config: dict[str, Any],
    num_users: int,
    mod_exists: bool,
) -> conversation_io.LLMConvData:
    assert num_users <= len(
        personas
    ), "Number of users must be less or equal to the number of provided personas"
    rand_personas = random.sample(personas, k=num_users)
    topic = random.choice(topics)

    user_names = [persona["username"] for persona in rand_personas]
    user_attributes = [format_persona_attributes(persona) for persona in rand_personas]

    data = conversation_io.LLMConvData(
        context=CTX_PREFACE + topic,
        user_names=user_names,
        user_attributes=user_attributes,
        user_instructions=user_instructions,
        moderator_name="moderator" if mod_exists else None,
        moderator_instructions=mod_instructions if mod_exists else None,
        moderator_attributes=[] if mod_exists else None,
        turn_manager_type=config["turn_manager_type"],
        turn_manager_config=config["turn_manager_config"],
        conv_len=config["conv_len"],
        history_ctx_len=config["history_ctx_len"],
    )
    return data


def format_persona_attributes(persona: dict[str, Any]) -> list[str]:
    attributes = []
    attributes.append(f"{persona["age"]} years old")
    attributes.append(persona["sexual_orientation"])
    attributes.append(persona["demographic_group"])
    attributes.append(persona["current_employment"])
    
    for characteristic in persona["personality_characteristics"]:
        attributes.append(characteristic)

    attributes.append(sex_parse(persona["sex"]))
    attributes.append(f"with {persona["education_level"]} education")
    attributes.append(f"and {persona["intent"]} intent")

    return attributes


def sex_parse(sex: str) -> str:
    match sex.lower():
        case "male":
            return "man"
        case "female":
            return "woman"
        case _:
            return "non-binary"

def main():
    parser = argparse.ArgumentParser(
        description="Generate conversation configs using modular configuration files"
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for generated conversation config files",
    )
    parser.add_argument(
        "--persona_dir",
        required=True,
        help="Directory containing JSON files for LLM user personas",
    )
    parser.add_argument(
        "--topics_dir",
        required=True,
        help="Directory containing .txt files for conversation starting comments",
    )
    parser.add_argument(
        "--configs_path",
        required=True,
        help="Path to JSON file containg conversation configs (such as conversation length)",
    )
    parser.add_argument(
        "--user_instruction_path",
        required=True,
        help="Path to .txt file containing user instructions",
    )
    parser.add_argument(
        "--mod_instruction_path",
        required=True,
        help="Path to .txt file containing moderator instructions",
    )
    parser.add_argument(
        "--num_generated_files",
        type=int,
        default=20,
        help="How many conversation files will be generated",
    )
    parser.add_argument(
        "--num_users",
        type=int,
        default=4,
        help="Number of users participating in the generated discussion",
    )
    parser.add_argument(
        "--include_mod",
        type=bool,
        default=True,
        help="Whether a moderator exists in the discussion",
    )

    args = parser.parse_args()

    print("Reading input files...")
    # Read files from the provided directories
    personas = read_files_from_directory(args.persona_dir, ".json")
    topics = read_files_from_directory(args.topics_dir, ".txt")
    user_instructions = read_file(args.user_instruction_path, ".txt")
    mod_instructions = read_file(args.mod_instruction_path, ".txt")
    config = read_file(args.configs_path, ".json")

    print("Processing...")
    discussion_io_objects = []
    for _ in range(args.num_generated_files):
        conv_file = generate_conv_config(
                personas=personas,
                topics=topics,
                user_instructions=user_instructions,
                mod_instructions=mod_instructions,
                config=config,
                num_users=args.num_users,
                mod_exists=args.include_mod,
            )
        discussion_io_objects.append(conv_file)
    
    print("Writing new conversation input files...")
    for io_object in discussion_io_objects:
        io_object.to_json_file(os.path.join(args.output_dir, str(uuid.uuid4())+".json"))
    print("Files exported to " + args.output_dir)

if __name__ == "__main__":
    main()
