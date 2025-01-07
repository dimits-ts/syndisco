import uuid
import os
import argparse
import shutil
import yaml
from pathlib import Path

from sdl.serialization import annotation_io, persona
from sdl.util.file_util import read_file, wipe_directory


def generate_annotator_file(
    annotator_persona: persona.LlmPersona,
    instructions: str,
    history_ctx_len: int,
    include_moderator_comments: bool,
) -> annotation_io.LlmAnnotationData:
    """Generate an annotation configuration object from provided attributes."""
    data = annotation_io.LlmAnnotationData(
        attributes=annotator_persona.to_attribute_list(),
        instructions=instructions,
        history_ctx_len=history_ctx_len,
        include_moderator_comments=include_moderator_comments,
    )
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate annotation files using configuration files.")
    parser.add_argument(
        "--config_file", 
        required=True, 
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Bypass the confirmation prompt and proceed with wiping files",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_file, "r") as file:
        config_data = yaml.safe_load(file)

    paths = config_data["generate_annotation_configs"]["paths"]
    experiment_vars = config_data["generate_annotation_configs"]["experiment_variables"]

    # Extract values from the config
    persona_dir = Path(paths["persona_dir"])
    annotation_export_dir = Path(paths["annotation_export_dir"])
    instruction_path = paths["instruction_path"]

    history_ctx_len = experiment_vars["history_ctx_len"]
    include_mod_comments = experiment_vars["include_mod_comments"]
    auto_confirm = args.yes

    # Ensure persona directory exists
    if not persona_dir.is_dir():
        print(f"Error: Persona directory '{persona_dir}' does not exist.")
        exit(1)

    # Ensure output directory exists or ask to wipe it
    if annotation_export_dir.is_dir():
        wipe_directory(annotation_export_dir, auto_confirm)
    else:
        os.makedirs(annotation_export_dir, exist_ok=True)

    print("Reading input files...")
    persona_files = os.listdir(persona_dir)
    personas = [
        persona.LlmPersona.from_json_file(os.path.join(persona_dir, persona_file))
        for persona_file in persona_files
    ]
    instructions = read_file(instruction_path)

    print("Processing...")
    for llm_persona in personas:
        annotation_config_file = generate_annotator_file(
            annotator_persona=llm_persona,
            instructions=instructions,
            history_ctx_len=history_ctx_len,
            include_moderator_comments=include_mod_comments
        )
        annotation_config_file.to_json_file(
            os.path.join(annotation_export_dir, str(uuid.uuid4()) + ".json")
        )
    
    print(f"Files exported to {annotation_export_dir}")


if __name__ == "__main__":
    main()