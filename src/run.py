"""
The main entry point for the library.
Imports the configuration file, sets up logging, and directs calls for discussion generation, 
annotation, and dataset export.
"""

import sys
import argparse
import logging
from pathlib import Path
import yaml
import pandas as pd

import sdl.util.logging_util
import sdl.util.file_util
import sdl.util.model_util
import sdl.annotations.experiments
import sdl.discussions.experiments
import sdl.postprocessing.postprocessing
import sdl.backend.model
from synthetic_discussion_framework.src.sdl.backend import actors
from synthetic_discussion_framework.src.sdl.discussions.experiments import (
    DiscussionExperiment,
)
from synthetic_discussion_framework.src.sdl.annotations.experiments import (
    AnnotationExperiment,
)


logger = logging.getLogger(Path(__file__).name)


def main():
    """
    Run synthetic discussion generation, annotation, and dataset export.

    This function parses the configuration file, sets up logging, and performs
    tasks based on the actions specified in the configuration. Actions include
    generating synthetic discussions, creating annotations, and exporting the dataset.
    """
    # Set up argument parser for config file path
    parser = argparse.ArgumentParser(description="Generate synthetic conversations")
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_file, "r", encoding="utf8") as file:
        yaml_data = yaml.safe_load(file)

    model_manager = sdl.util.model_util.ModelManager(yaml_data=yaml_data)

    generate_discussions = yaml_data["actions"]["generate_discussions"]
    generate_annotations = yaml_data["actions"]["generate_annotations"]
    export_dataset = yaml_data["actions"]["export_dataset"]

    setup_logging(logging_config=yaml_data["logging"])
    validate_actions(
        generate_discussions=generate_discussions,
        generate_annotations=generate_annotations,
        export_dataset=export_dataset,
    )
    if generate_discussions:
        discussion_exp = create_discussion_experiment(
            llm=model_manager.get(), 
            discussion_config=yaml_data["discussions"]
        )
        run_discussion_experiment(
            experiment=discussion_exp,
            output_dir=yaml_data["discussions"]["files"]["output_dir"],
        )

    if generate_annotations:
        ann_exp = create_annotation_experiment(
            llm=model_manager.get(), 
            annotation_config=yaml_data["annotations"]
        )
        run_annotation_experiment(
            ann_exp,
            discussions_dir=yaml_data["discussions"]["files"]["output_dir"],
            output_dir=yaml_data["annotation"]["output_dir"],
        )

    if export_dataset:
        dataset_to_csv(yaml_data["dataset_export"])


def setup_logging(logging_config: dict) -> None:
    sdl.util.logging_util.logging_setup(
        print_to_terminal=logging_config["print_to_terminal"],
        write_to_file=logging_config["write_to_file"],
        logs_dir=Path(logging_config["logs_dir"]),
        level=logging_config["level"],
        use_colors=True,
        log_warnings=True,
    )


def validate_actions(
    generate_discussions, generate_annotations, export_dataset
) -> None:
    if not generate_discussions and not generate_annotations and not export_dataset:
        logger.warning("All procedures have been disabled for this run. Exiting...")
        sys.exit(0)
    else:
        if not generate_discussions:
            logger.warning("Synthetic discussion generation disabled.")
        if not generate_annotations:
            logger.warning("Synthetic annotation disabled.")
        if not export_dataset:
            logger.warning("Dataset export to CSV disabled.")


def create_discussion_experiment(llm, discussion_config: dict) -> DiscussionExperiment:
    topics = sdl.util.file_util.read_files_from_directory(
        discussion_config["files"]["topics_dir"]
    )

    users = actors.create_users_from_file(
        llm,
        persona_path=discussion_config["files"]["user_persona_path"],
        instruction_path=discussion_config["files"]["user_instructions_path"],
        context=discussion_config["experiment_variables"]["context_prompt"],
        actor_type=actors.ActorType.USER,
    )

    if discussion_config["experiment_variables"]["include_mod"]:
        mod_instructions = sdl.util.file_util.read_file(
            discussion_config["files"]["mod_instruction_path"]
        )
        moderator = actors.create_users(
            llm,
            usernames=["moderator"],
            attributes=[
                discussion_config["experiment_variables"]["moderator_attributes"]
            ],
            context=discussion_config["experiment_variables"]["context_prompt"],
            actor_type=actors.ActorType.USER,
            instructions=mod_instructions,
        )[0]
    else:
        moderator = None

    return DiscussionExperiment(
        topics=topics,
        users=users,
        moderator=moderator,
        num_turns=discussion_config["experiment_variables"]["num_experiments"],
        num_active_users=discussion_config["experiment_variables"]["num_users"],
        num_discussions=discussion_config["experiment_variables"]["num_experiments"],
    )


def run_discussion_experiment(
    experiment: DiscussionExperiment, output_dir: Path
) -> None:
    logger.info("Starting synthetic discussion experiments...")
    experiment.begin(discussions_output_dir=output_dir)
    logger.info("Finished synthetic discussion experiments.")


def create_annotation_experiment(llm, annotation_config: dict) -> AnnotationExperiment:
    annotators = actors.create_users_from_file(
        llm,
        persona_path=annotation_config["files"]["annotator_persona_path"],
        instruction_path=annotation_config["files"]["instruction_path"],
        context="You are a human annotator",
        actor_type=actors.ActorType.ANNOTATOR,
    )

    return AnnotationExperiment(
        annotators=annotators,
        history_ctx_len=annotation_config["experiment_variables"]["history_ctx_len"],
        include_mod_comments=annotation_config["experiment_variables"][
            "include_mod_comments"
        ],
    )


def run_annotation_experiment(
    annotation_experiment: AnnotationExperiment, discussions_dir: Path, output_dir: Path
) -> None:
    logger.info("Starting synthetic annotation...")
    annotation_experiment.begin(discussions_dir, output_dir)
    logger.info("Finished synthetic annotation.")


def dataset_to_csv(export_config) -> None:
    conv_dir = Path(export_config["discussion_root_dir"])
    annot_dir = Path(export_config["annotation_root_dir"])
    export_path = Path(export_config["export_path"])

    df = _create_dataset(conv_dir=conv_dir, annot_dir=annot_dir)
    _export_dataset(df=df, output_path=export_path)
    logger.info(f"Dataset exported to {export_path}")


def _create_dataset(conv_dir: Path, annot_dir: Path) -> pd.DataFrame:
    """
    Create a combined dataset from conversation and annotation files.

    :param conv_dir: Directory containing conversation data files.
    :type conv_dir: Path
    :param annot_dir: Directory containing annotation data files.
    :type annot_dir: Path
    :return: A combined DataFrame containing conversation and annotation data.
    :rtype: pd.DataFrame
    """
    conv_df = sdl.postprocessing.postprocessing.import_conversations(conv_dir)
    conv_df = conv_df.rename({"id": "conv_id"}, axis=1)
    annot_df = sdl.postprocessing.postprocessing.import_annotations(annot_dir)

    full_df = pd.merge(
        left=conv_df,
        right=annot_df,
        on=["conv_id", "message"],
        how="left",
        suffixes=["_conv", "_annot"],  # type: ignore
    )
    del full_df["index_annot"]
    del full_df["index_conv"]

    return full_df


def _export_dataset(df: pd.DataFrame, output_path: Path):
    """
    Export the dataset to a CSV file.

    :param df: The dataset to export.
    :type df: pd.DataFrame
    :param output_path: The path where the exported dataset will be saved.
    :type output_path: Path
    """
    sdl.util.file_util.ensure_parent_directories_exist(output_path)
    df.to_csv(
        path_or_buf=output_path, encoding="utf8", mode="w+"
    )  # overwrite previous dataset


if __name__ == "__main__":
    main()
