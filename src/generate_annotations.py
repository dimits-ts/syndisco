import argparse
import os
import yaml
from pathlib import Path

from sdl.serialization import annotation_io
from sdl.util import file_util
from sdl.backend import model


REMOVE_STR_LIST = ["```"]


def process_file(
    annotation_config_input_file: str | Path,
    output_dir: str | Path,
    model: model.Model,
    conv_logs_path: str | Path,
) -> None:
    print(f"Processing file: {annotation_config_input_file}")

    # Load data and start conversation
    data = annotation_io.LlmAnnotationData.from_json_file(annotation_config_input_file)
    generator = annotation_io.LLMAnnotationGenerator(
        data=data, llm=model, conv_logs_path=conv_logs_path
    )
    conv = generator.produce_conversation()

    print("Beginning conversation...")
    conv.begin_annotation(verbose=True)
    output_path = file_util.generate_datetime_filename(
        output_dir=output_dir, file_ending=".json"
    )
    conv.to_json_file(output_path)
    print("Conversation saved to ", output_path)


def main():
    # Set up argument parser for config file path
    parser = argparse.ArgumentParser(description="Generate synthetic annotations")
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config_file, "r") as file:
        config_data = yaml.safe_load(file)

    paths = config_data["generate_annotations"]["paths"]
    model_params = config_data["generate_annotations"]["model_parameters"]

    # Extract values from the config
    input_dir = Path(paths["input_dir"])
    output_dir = Path(paths["output_dir"])
    prompt_path = Path(paths["instruction_path"])
    model_path = paths["model_path"]
    convs_dir = Path(paths["conv_logs_dir"])

    library_type = model_params["library_type"]
    model_name = model_params["model_name"]
    max_tokens = model_params["general"]["max_tokens"]
    ctx_width_tokens = model_params["general"]["ctx_width_tokens"]
    inference_threads = model_params["llama_cpp"]["inference_threads"]
    gpu_layers = model_params["llama_cpp"]["gpu_layers"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if input directory exists
    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        exit(1)

    # Check if prompt file exists
    if not prompt_path.is_file():
        print(f"Error: Prompt file '{prompt_path}' does not exist.")
        exit(1)

    if not convs_dir.is_dir():
        print(f"Error: Convs directory '{convs_dir}' does not exist.")
        exit(1)

    # Load model based on type
    print("Loading LLM model...")

    model = None
    if library_type == "llama_cpp":
        # dynamically load library to avoid dependency hell
        from sdl.backend.cpp_model import LlamaModel 

        model = LlamaModel(
            model_path=model_path,
            name=model_name,
            max_out_tokens=max_tokens,
            seed=42,  # Random seed (this can be adjusted)
            remove_string_list=REMOVE_STR_LIST,
            ctx_width_tokens=ctx_width_tokens,
            inference_threads=inference_threads,
            gpu_layers=gpu_layers,
        )
    elif library_type == "transformers":
        # dynamically load library to avoid dependency hell
        from sdl.backend.trans_model import TransformersModel

        model = TransformersModel(
            model_path=model_path,
            name=model_name,
            max_out_tokens=max_tokens,
            remove_string_list=REMOVE_STR_LIST,
        )
    else:
        raise NotImplementedError(
            f"Unknown model type: {library_type}. Supported types: llama_cpp, transformers"
        )

    print("Model loaded.")

    # Process the files in the input directory
    print(f"Starting annotation generation...")

    for completed_discussion_path in convs_dir.iterdir():
        for persona_input_file in input_dir.glob("*.json"):
            if persona_input_file.is_file():
                process_file(persona_input_file, output_dir, model, completed_discussion_path)
            else:
                print(f"Skipping non-file entry: {persona_input_file}")

    print(f"Finished annotation generation.")


if __name__ == "__main__":
    main()
