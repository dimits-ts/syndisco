import os
import logging
from pathlib import Path

from sdl.backend import persona, model, actors
from ..util import file_util
from . import generation


logger = logging.getLogger(Path(__file__).name)


def run_experiments(
    llm: model.Model,
    yaml_data: dict
) -> None:
    discussions_dir = Path(yaml_data["discussions"]["files"]["output_dir"])

    annotation_file_data = yaml_data["annotation"]["files"]
    annotator_persona_dir = Path(annotation_file_data["annotator_persona_dir"])
    output_dir = Path(annotation_file_data["output_dir"])
    instruction_path = Path(annotation_file_data["instruction_path"])

    annotation_experiment_data = yaml_data["annotation"]["experiment_variables"]
    include_mod_comments = annotation_experiment_data["include_mod_comments"]
    history_ctx_len = annotation_experiment_data["history_ctx_len"]

    # Ensure annotation config output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not discussions_dir.is_dir():
        logger.error(f"Error: Synthetic discussion directory '{discussions_dir}' does not exist.")
        exit(1)


    annotation_experiments = _generate_experiments(
        llm=llm,
        persona_dir=annotator_persona_dir,
        instruction_path=instruction_path,
        discussions_dir=discussions_dir,
        history_ctx_len=history_ctx_len,
        include_mod_comments=include_mod_comments,
    )

    for annotation_experiment in annotation_experiments:
        _run_single_experiment(annotation_experiment, output_dir)

    logger.info("Finished annotation generation.")


def _run_single_experiment(experiment: generation.AnnotationConv, output_dir: Path) -> None:
    try:
        logger.info("Beginning conversation...")
        experiment.begin_annotation(verbose=True)
        output_path = file_util.generate_datetime_filename(
            output_dir=output_dir, file_ending=".json"
        )
        experiment.to_json_file(output_path)
        logger.info("Annotation saved to " + str(output_path))
    except Exception:
        logger.exception("Experiment aborted due to error.")


def _generate_annotator_conv(
    llm: model.Model,
    conv_logs_path: str | Path,
    annotator_persona: persona.LlmPersona,
    instructions: str,
    history_ctx_len: int,
    include_moderator_comments: bool,
) -> generation.AnnotationConv:
    """Generate an annotation configuration object from provided attributes."""
    annotator = actors.LLMAnnotator(
        model=llm,
        name="annotator",
        attributes=annotator_persona.to_attribute_list(),
        context="You are a human working as an annotator",
        instructions=instructions,
    )

    conversation = generation.AnnotationConv(
        annotator=annotator,
        conv_logs_path=conv_logs_path,
        history_ctx_len=history_ctx_len,
        include_moderator_comments=include_moderator_comments,
    )
    return conversation


def _generate_experiments(
    llm: model.Model,
    persona_dir: Path,
    instruction_path: Path,
    discussions_dir: Path,
    history_ctx_len: int,
    include_mod_comments: bool,
) -> list[generation.AnnotationConv]:

    # Ensure persona directory exists
    if not persona_dir.is_dir():
        logger.error(f"Error: Persona directory '{persona_dir}' does not exist.")
        exit(1)
    
    if not instruction_path.exists():
        logger.error(f"Error: Instructions file '{instruction_path}' does not exist.")
        exit(1)

    if not discussions_dir.is_dir():
        logger.error(f"Error: Discussions directory '{discussions_dir}' does not exist.")
        exit(1)

    persona_files = os.listdir(persona_dir)
    personas = [
        persona.LlmPersona.from_json_file(os.path.join(persona_dir, persona_file))
        for persona_file in persona_files
    ]
    instructions = file_util.read_file(instruction_path)

    annotation_experiments = []
    for llm_persona in personas:
        for discussion_path in discussions_dir.iterdir():
            annotation_experiment = _generate_annotator_conv(
                llm=llm,
                conv_logs_path=discussion_path,
                annotator_persona=llm_persona,
                instructions=instructions,
                history_ctx_len=history_ctx_len,
                include_moderator_comments=include_mod_comments,
            )
            annotation_experiments.append(annotation_experiment)

    return annotation_experiments
