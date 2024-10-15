#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT_DIR/src"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT_DIR="$(dirname "$SRC_DIR")"

bash "$SCRIPT_DIR/conversation_execute_all.sh" \
    --python_script_path  "$SRC_DIR/sdf_create_conversations.py" \
    --input_dir "$PROJECT_ROOT_DIR/data/generated_discussions_input" \
    --output_dir "$PROJECT_ROOT_DIR/data/generated_discussions_output" \
    --model_path "$SRC_DIR/models/llama-3-8B.gguf"