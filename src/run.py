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

    # Set up logging config
    logging_config = yaml_data["logging"]
    sdl.util.logging_util.logging_setup(
        print_to_terminal=logging_config["print_to_terminal"],
        write_to_file=logging_config["write_to_file"],
        logs_dir=Path(logging_config["logs_dir"]),
        level=logging_config["level"],
        use_colors=True,
        log_warnings=True,
    )

    action_config = yaml_data["actions"]
    generate_discussions = action_config["generate_discussions"]
    generate_annotations = action_config["generate_annotations"]
    export_dataset = action_config["export_dataset"]

    model_manager = sdl.util.model_util.ModelManager(yaml_data=yaml_data)

    # Disabled functionality warnings
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

    if generate_discussions:
        # Create discussions
        llm = model_manager.get()
        logger.info("Starting synthetic discussion experiments...")
        sdl.discussions.experiments.run_experiments(llm=llm, yaml_data=yaml_data)
        logger.info("Finished synthetic discussion experiments.")

    if generate_annotations:
        # Create annotations
        llm = model_manager.get()
        logger.info("Starting synthetic annotation...")
        sdl.annotations.experiments.run_experiments(llm=llm, yaml_data=yaml_data)
        logger.info("Finished synthetic annotation.")

    if export_dataset:
        # Export full dataset
        export_config = yaml_data["dataset_export"]
        conv_dir = Path(export_config["discussion_root_dir"])
        annot_dir = Path(export_config["annotation_root_dir"])
        export_path = Path(export_config["export_path"])

        df = _create_dataset(conv_dir=conv_dir, annot_dir=annot_dir)
        _export_dataset(df=df, output_path=export_path)
        logger.info(f"Dataset exported to {export_path}")


def _create_dataset(
    conv_dir: Path, annot_dir: Path
) -> pd.DataFrame:
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
    annot_df = sdl.postprocessing.postprocessing.import_annotations(
        annot_dir
    )

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
