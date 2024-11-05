#!/bin/bash

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SCRIPT_PATH="$PROJECT_ROOT_DIR/generate_annotations.py"

usage() {
  echo "Usage: $0 --python_script_path <python script path> --conv_input_dir <input_directory> --prompt_path <input_path> --output_dir <output_directory> --model_path <model_file_path>"
  exit 1
}


while [[ "$#" -gt 0 ]]; do
  case $1 in
    --conv_input_dir) conv_input_dir="$2"; shift ;;
    --prompt_path) prompt_path="$2"; shift ;;
    --output_dir) output_dir="$2"; shift ;;
    --model_path) model_path="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

# Check if all required arguments are provided
if [[ -z "$conv_input_dir" || -z "$prompt_path" || -z "$output_dir" || -z "$model_path" ]]; then
  echo "Error: Missing required arguments."
  usage
fi

if [[ ! -d "$conv_input_dir" ]]; then
  echo "Error: Conversation input directory '$conv_input_dir' does not exist."
  exit 1
fi

if [[ ! -f "$prompt_path" ]]; then
  echo "Error: Annotation prompt input path '$prompt_path' does not exist."
  exit 1
fi

if [[ ! -f "$model_path" ]]; then
  echo "Error: Model file path '$model_path' does not exist."
  exit 1
fi

for input_file in "$conv_input_dir"/*; do
  if [[ -f "$input_file" ]]; then
    echo "Processing file: $input_file"
    python -u "$SCRIPT_PATH" --prompt_input_path="$prompt_path" --conv_path="$input_file" --output "$output_dir" --model_path "$model_path" --ctx_width_tokens=2048 --gpu_layers 8
  else
    echo "Skipping non-file entry: $input_file"
  fi
done
