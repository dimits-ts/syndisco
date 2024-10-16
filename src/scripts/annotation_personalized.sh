#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT_DIR="$(dirname "$SRC_DIR")"

PYTHON_SCRIPT_PATH="$SCRIPT_DIR/sdf_create_annotations.py"
OUTPUT_DIR="$PROJECT_ROOT_DIR/data/annotations_output"
MODEL_PATH="$SRC_DIR/models/llama-3-8B.gguf"
CONV_INPUT_DIR="$PROJECT_ROOT_DIR/generated_discussions_output"
ANNOTATOR_PROMPT_DIR="$PROJECT_ROOT_DIR/data/annotations_input"


# for each synthetic conversation
for DIR in "$CONV_INPUT_DIR"/*; do
    # for each annotator SDB prompt
    for ANNOTATOR_PROMPT_PATH in "$ANNOTATOR_PROMPT_DIR"/*; do
        bash "$SCRIPT_DIR/annotation_execute_all.sh" \
        --python_script_path "$PYTHON_SCRIPT_PATH" \
        --conv_input_dir "$DIR" \
        --prompt_path "$ANNOTATOR_PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH"
    done
done
