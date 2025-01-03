import uuid
import os
import random
import yaml
import argparse
from typing import Any
from pathlib import Path

from sdl.serialization.persona import LlmPersona
from sdl.serialization import conversation_io
from sdl.util.file_util import read_files_from_directory, read_file, read_json_file


CONTEXT = "You are a human participating in an online chatroom."
DEFAULT_MODERATOR_ATTRIBUTES = ["just", "strict", "understanding"]
SEED_USERNAMES = ["FirstUser"]


def generate_conv_config(
    personas: list[LlmPersona],
    user_instructions: str,
    mod_instructions: str,
    seed_opinions: list[str],
    seed_opinion_usernames: list[str],
    config: dict[str, Any],
    num_users: int,
    mod_exists: bool,
) -> conversation_io.LLMConvData:
    """Generate a conversation configuration object from provided attributes."""
    assert num_users <= len(personas), "Number of users must be less or equal to the number of provided personas"
    rand_personas = random.sample(personas, k=num_users)
    topic = random.choice(seed_opinions)
    seed_username = random.choice(seed_opinion_usernames)

    user_names = [persona.username for persona in rand_personas]
    user_attributes = [persona.to_attribute_list() for persona in rand_personas]

    data = conversation_io.LLMConvData(
        context=CONTEXT,
        user_names=user_names,
        user_attributes=user_attributes,
        user_instructions=user_instructions,
        moderator_name="moderator" if mod_exists else None,
        moderator_instructions=mod_instructions if mod_exists else None,
        moderator_attributes=DEFAULT_MODERATOR_ATTRIBUTES if mod_exists else None,
        turn_manager_type=config["turn_manager_type"],
        turn_manager_config=config["turn_manager_config"],
        conv_len=config["conv_len"],
        history_ctx_len=config["history_ctx_len"],
        seed_opinions=[topic],  # only one seed opinion for our experiments
        seed_opinion_usernames=[seed_username]  # only one seed opinion for our experiments
    )
    return data


def main():
    # Set up argument parsing for the YAML config file
    parser = argparse.ArgumentParser(description="Generate conversation configurations")
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Bypass the confirmation prompt and proceed with wiping files",
    )
    args = parser.parse_args()

    # Load the provided YAML configuration file
    with open(args.config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract yaml configs
    paths = config_data["generate_conv_configs"]["paths"]
    experiment_variables = config_data["generate_conv_configs"]["experiment_variables"]

    # Paths for various required files and directories
    topics_dir = paths["topics_dir"]
    persona_dir = paths["persona_dir"]
    user_instruction_path = paths["user_instructions_path"]
    mod_instruction_path = paths["mod_instructions_path"]
    turn_taking_configs_path = paths["turn_taking_configs_path"]
    data_output_dir = Path(paths["experiment_export_dir"])

    # Experiment variables
    num_generated_files = experiment_variables["num_generated_files"]
    num_users = experiment_variables["num_users"]
    include_mod = experiment_variables["include_mod"]

   # Ask for confirmation before wiping files
    confirmation = input(f"Are you sure you want to remove all files in {data_output_dir}? [y/n]: ").strip().lower()

    if confirmation == 'y' or args.yes:
        print("Removing old generated files...")
        if data_output_dir.exists() and data_output_dir.is_dir():
            for file in data_output_dir.iterdir():
                if file.is_file():
                    file.unlink()
    else:
        print("Skipping the removal of old files.")

    # Ensure the output directory exists 
    os.makedirs(data_output_dir, exist_ok=True)

    print("Reading input files...")
    persona_files = os.listdir(persona_dir)
    personas = [
        LlmPersona.from_json_file(os.path.join(persona_dir, persona_file))
        for persona_file in persona_files
    ]

    topics = read_files_from_directory(topics_dir)
    user_instructions = read_file(user_instruction_path)
    mod_instructions = read_file(mod_instruction_path)
    config = read_json_file(turn_taking_configs_path)

    print("Processing...")
    discussion_io_objects = []
    for _ in range(num_generated_files):
        conv_file = generate_conv_config(
            personas=personas,
            user_instructions=user_instructions,
            mod_instructions=mod_instructions,
            config=config,
            num_users=num_users,
            mod_exists=include_mod,
            seed_opinions=topics,
            seed_opinion_usernames=SEED_USERNAMES
        )
        discussion_io_objects.append(conv_file)

    print("Writing new conversation input files...")
    for io_object in discussion_io_objects:
        io_object.to_json_file(
            os.path.join(data_output_dir, str(uuid.uuid4()) + ".json")
        )
    print("Files exported to " + str(data_output_dir))


if __name__ == "__main__":
    main()
