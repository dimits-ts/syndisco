#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SRC_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT_DIR="$(dirname "$SRC_DIR")"
LOG_DIR="$PROJECT_ROOT_DIR/logs"
CURRENT_DATE=$(date +'%Y-%m-%d')

mkdir -p "$LOG_DIR"

bash "$SCRIPT_DIR/conversation_execute_all.sh" \
    --python_script_path  "$SRC_DIR/sdf_create_conversations.py" \
    --input_dir "$PROJECT_ROOT_DIR/data/generated_discussions_input/conv_data/generated" \
    --output_dir "$PROJECT_ROOT_DIR/data/generated_discussions_output/vanilla_no_mod" \
    --model_path "$SRC_DIR/models/llama-3-8B-instruct.gguf" \
    --max_tokens 500 \
    --ctx_width_tokens 2048 \
    --inference_threads 10 \
    --gpu_layers 10  \
    2>&1 | tee -a "$LOG_DIR/$CURRENT_DATE.txt"