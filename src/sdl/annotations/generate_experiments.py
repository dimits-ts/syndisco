import uuid
import os
import logging
from pathlib import Path

from sdl.backend import persona
from sdl.util.file_util import read_file, wipe_directory
import sdl.annotations.io


logger = logging.getLogger(__name__)


def generate_annotator_file(
    annotator_persona: persona.LlmPersona,
    instructions: str,
    history_ctx_len: int,
    include_moderator_comments: bool,
) -> sdl.annotations.io.LlmAnnotationData:
    """Generate an annotation configuration object from provided attributes."""
    data = sdl.annotations.io.LlmAnnotationData(
        attributes=annotator_persona.to_attribute_list(),
        instructions=instructions,
        history_ctx_len=history_ctx_len,
        include_moderator_comments=include_moderator_comments,
    )
    return data


def generate_experiments(yaml_data: dict, auto_confirm: bool):
    paths = yaml_data["generate_annotation_configs"]["paths"]
    experiment_vars = yaml_data["generate_annotation_configs"]["experiment_variables"]

    # Extract values from the config
    persona_dir = Path(paths["persona_dir"])
    annotation_export_dir = Path(paths["annotation_export_dir"])
    instruction_path = paths["instruction_path"]

    history_ctx_len = experiment_vars["history_ctx_len"]
    include_mod_comments = experiment_vars["include_mod_comments"]

    # Ensure persona directory exists
    if not persona_dir.is_dir():
        logger.error(f"Error: Persona directory '{persona_dir}' does not exist.")
        exit(1)

    # Ensure output directory exists or ask to wipe it
    if annotation_export_dir.is_dir():
        wipe_directory(annotation_export_dir, auto_confirm)
    else:
        os.makedirs(annotation_export_dir, exist_ok=True)

    logger.info("Reading input files...")
    persona_files = os.listdir(persona_dir)
    personas = [
        persona.LlmPersona.from_json_file(os.path.join(persona_dir, persona_file))
        for persona_file in persona_files
    ]
    instructions = read_file(instruction_path)

    logger.info("Processing...")
    for llm_persona in personas:
        annotation_config_file = generate_annotator_file(
            annotator_persona=llm_persona,
            instructions=instructions,
            history_ctx_len=history_ctx_len,
            include_moderator_comments=include_mod_comments,
        )
        annotation_config_file.to_json_file(
            os.path.join(annotation_export_dir, str(uuid.uuid4()) + ".json")
        )

    logger.info(f"Files exported to {annotation_export_dir}")
