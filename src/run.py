import argparse
import logging
from pathlib import Path
import yaml

import pandas as pd

import sdl.util.logging_util
import sdl.util.file_util
import sdl.annotations.experiments
import sdl.discussions.experiments
import sdl.postprocessing.postprocessing
import sdl.backend.model


logger = logging.getLogger(__name__)


def main():
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
        logs_dir=logging_config["logs_dir"],
        level=logging_config["level"],
    )

    # Load model
    logger.info("Loading LLM...")
    llm = _initialize_model(yaml_data)
    logger.info("Model loaded.")

    # Create discussions
    logger.info("Starting synthetic discussion experiments...")
    sdl.discussions.experiments.run_experiments(llm=llm, yaml_data=yaml_data)
    logger.info("Finished synthetic discussion experiments.")

    # Create annotations
    logger.info("Starting synthetic annotation...")
    sdl.annotations.experiments.run_experiments(llm=llm, yaml_data=yaml_data)
    logger.info("Finished synthetic annotation.")

    # Export full dataset
    conv_dir = yaml_data["discussions"]["files"]["output_dir"]
    annot_dir = yaml_data["annotation"]["files"]["output_dir"]
    export_path = yaml_data["dataset_export"]["export_path"]
    include_sdbs = yaml_data["dataset_export"]["include_annotator_sdb_info"]

    df = _create_dataset(
        conv_dir=conv_dir, 
        annot_dir=annot_dir, 
        include_sdbs=include_sdbs
    )
    _export_dataset(df=df, output_path=export_path)
    logger.info("Dataset exported to " + str(export_path))


def _initialize_model(yaml_data: dict) -> sdl.backend.model.Model:
    # Extract values from the config
    model_params = yaml_data["model_parameters"]
    model_path = model_params["general"]["model_path"]
    model_name = model_params["general"]["model_pseudoname"]
    library_type = model_params["general"]["library_type"]
    max_tokens = model_params["general"]["max_tokens"]
    ctx_width_tokens = model_params["general"]["ctx_width_tokens"]
    remove_str_list = model_params["general"]["disallowed_strings"]

    inference_threads = model_params["llama_cpp"]["inference_threads"]
    gpu_layers = model_params["llama_cpp"]["gpu_layers"]

    llm = None
    if library_type == "llama_cpp":
        # dynamically load library to avoid dependency hell
        from sdl.backend.cpp_model import LlamaModel

        llm = LlamaModel(
            model_path=model_path,
            name=model_name,
            max_out_tokens=max_tokens,
            seed=42,  # Random seed (this can be adjusted)
            remove_string_list=remove_str_list,
            ctx_width_tokens=ctx_width_tokens,
            inference_threads=inference_threads,
            gpu_layers=gpu_layers,
        )
    elif library_type == "transformers":
        # dynamically load library to avoid dependency hell
        from sdl.backend.trans_model import TransformersModel

        llm = TransformersModel(
            model_path=model_path,
            name=model_name,
            max_out_tokens=max_tokens,
            remove_string_list=remove_str_list,
        )
    else:
        raise NotImplementedError(
            f"Unknown model type: {library_type}. Supported types: llama_cpp, transformers"
        )
    return llm


def _create_dataset(
    conv_dir: Path, annot_dir: Path, include_sdbs: bool
) -> pd.DataFrame:
    if not include_sdbs:
        logger.warning("Not including annotator SDBs to output.")

    conv_df = sdl.postprocessing.postprocessing.import_conversations(conv_dir)
    conv_df = conv_df.rename({"id": "conv_id"}, axis=1)
    annot_df = sdl.postprocessing.postprocessing.import_annotations(
        annot_dir, include_sdbs
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
    sdl.util.file_util.ensure_parent_directories_exist(output_path)
    df.to_csv(
        path_or_buf=output_path, 
        encoding="utf8", 
        mode="w+"
    )  # overwrite previous dataset


if __name__ == "__main__":
    main()
