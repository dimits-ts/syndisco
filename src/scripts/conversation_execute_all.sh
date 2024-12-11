#!/bin/bash

THIS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$THIS_DIR")"
SCRIPT_PATH="$PROJECT_ROOT_DIR/generate_conversations.py"

usage() {
  echo "Usage: $0 --python_script_path <python script path> --input_dir <input_directory> --output_dir <output_directory> --model_path <model_file_path> --type <model_type> --model_name <model_name>"
  exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --input_dir) input_dir="$2"; shift ;;
    --output_dir) output_dir="$2"; shift ;;
    --model_path) model_path="$2"; shift ;;
    --ctx_width_tokens) ctx_width_tokens="$2"; shift ;; 
    --gpu_layers) gpu_layers="$2"; shift ;; 
    --max_tokens) max_tokens="$2"; shift ;; 
    --inference_threads) inference_threads="$2"; shift ;;
    --type) model_type="$2"; shift ;; 
    --model_name) model_name="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; usage ;;
  esac
  shift
done

# Check if all required arguments are provided
if [[ -z "$input_dir" || -z "$output_dir" || -z "$model_path" || -z "$model_type" || -z "$model_name" ]]; then
  echo "Error: Missing required arguments."
  usage
fi

if [[ ! -d "$input_dir" ]]; then
  echo "Error: Input directory '$input_dir' does not exist."
  exit 1
fi

# Process each input file
for input_file in "$input_dir"/*; do
  if [[ -f "$input_file" ]]; then
    echo "Processing file: $input_file"
    python -u "$SCRIPT_PATH" \
          --output_dir "$output_dir" \
          --model_path "$model_path" \
          --input_file "$input_file" \
          --ctx_width_tokens "$ctx_width_tokens" \
          --max_tokens "$max_tokens" \
          --gpu_layers "$gpu_layers" \
          --inference_threads "$inference_threads" \
          --type "$model_type" \
          --model_name "$model_name"
  else
    echo "Skipping non-file entry: $input_file"
  fi
done
