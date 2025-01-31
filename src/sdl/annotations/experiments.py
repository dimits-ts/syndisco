"""
Generates an annotation task for each annotator and for each synthetic discussion.  
Each annotation task is packaged into an AnnotationConv object (@see generation.py).
Then runs each annotation task sequentially, and saves the output to disk as an auto-generated file.
"""
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
    """
    Creates annotation experiments for synthetic discussions and saves their outputs.

    :param llm: The language model instance.
    :type llm: model.Model
    :param yaml_data: Serialized YAML configuration containing input paths and parameters.
    :type yaml_data: dict
    """
    discussions_dir = Path(yaml_data["discussions"]["files"]["output_dir"])

    annotation_file_data = yaml_data["annotation"]["files"]
    annotator_persona_path = Path(annotation_file_data["annotator_persona_path"])
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
        persona_path=annotator_persona_path,
        instruction_path=instruction_path,
        discussions_dir=discussions_dir,
        history_ctx_len=history_ctx_len,
        include_mod_comments=include_mod_comments,
    )

    for i, annotation_experiment in enumerate(annotation_experiments):
        logging.info(f"Running experiment {i}/{len(annotation_experiments)}...")
        _run_single_experiment(annotation_experiment, output_dir)

    logger.info("Finished annotation generation.")


def _run_single_experiment(experiment: generation.AnnotationConv, output_dir: Path) -> None:
    """
    Executes a single annotation experiment and saves its output to a file.

    :param experiment: An AnnotationConv object containing the annotation task.
    :type experiment: generation.AnnotationConv
    :param output_dir: The directory to save the experiment's output file.
    :type output_dir: Path
    """
    try:
        logger.info("Beginning conversation...")
        logger.debug(f"Experiment parameters: {str(experiment)}")
        experiment.begin_annotation(verbose=True)
        output_path = file_util.generate_datetime_filename(
            output_dir=output_dir, file_ending=".json"
        )
        experiment.to_json_file(output_path)
        logger.info(f"Annotation saved to {output_path}")
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
    """
    Generates a single annotation task configuration.

    :param llm: The language model instance used for the annotator.
    :type llm: model.Model
    :param conv_logs_path: Path to the conversation logs for the annotation.
    :type conv_logs_path: str | Path
    :param annotator_persona: Persona details of the annotator.
    :type annotator_persona: persona.LlmPersona
    :param instructions: Instructions provided to the annotator.
    :type instructions: str
    :param history_ctx_len: Context history length provided to the annotator.
    :type history_ctx_len: int
    :param include_moderator_comments: Flag indicating whether moderator comments are included.
    :type include_moderator_comments: bool
    :return: A configured AnnotationConv object.
    :rtype: generation.AnnotationConv
    """
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
    persona_path: Path,
    instruction_path: Path,
    discussions_dir: Path,
    history_ctx_len: int,
    include_mod_comments: bool,
) -> list[generation.AnnotationConv]:
    """
    Generates a list of annotation experiments by combining synthetic discussions
    and annotator personas.

    :param llm: The language model instance.
    :type llm: model.Model
    :param persona_path: The JSON file containing annotator personas.
    :type persona_dir: Path
    :param instruction_path: Path to the instructions file for annotators.
    :type instruction_path: Path
    :param discussions_dir: Directory containing synthetic discussion files.
    :type discussions_dir: Path
    :param history_ctx_len: Context history length provided to annotators.
    :type history_ctx_len: int
    :param include_mod_comments: Flag indicating whether moderator comments are included.
    :type include_mod_comments: bool
    :return: List of configured AnnotationConv objects for annotation experiments.
    :rtype: list[generation.AnnotationConv]

    :raises SystemExit: If any required directory or file is missing.
    """
    # Ensure persona directory exists
    if not persona_path.exists():
        logger.error(f"Error: Persona file '{persona_path}' does not exist.")
        exit(1)

    if not instruction_path.exists():
        logger.error(f"Error: Instructions file '{instruction_path}' does not exist.")
        exit(1)

    if not discussions_dir.is_dir():
        logger.error(f"Error: Discussions directory '{discussions_dir}' does not exist.")
        exit(1)

    personas = persona.from_json_file(persona_path)
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