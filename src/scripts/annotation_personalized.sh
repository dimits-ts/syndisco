#!/bin/bash

PYTHON_SCRIPT_PATH="/home/dimits/Documents/university/research/llm_moderation/experiments/llm_annotation.py"
OUTPUT_DIR="/home/dimits/Documents/university/research/llm_moderation/experiments/output/annotations"
MODEL_PATH="/home/dimits/Documents/university/research/llm_moderation/experiments/models/llama-2-13b-chat.Q5_K_M.gguf"
CONV_INPUT_DIR="/home/dimits/Documents/university/research/llm_moderation/experiments/output/conversations"

declare -a ANNOTATOR_PROMPT_PATHS=(
    "/home/dimits/Documents/university/research/llm_moderation/experiments/data/annotations/annot_blue_collar.json"
    "/home/dimits/Documents/university/research/llm_moderation/experiments/data/annotations/annot_gamer.json"
    "/home/dimits/Documents/university/research/llm_moderation/experiments/data/annotations/annot_grandma.json"
    "/home/dimits/Documents/university/research/llm_moderation/experiments/data/annotations/annot_professor.json"
)

# Loop through all files in the specified directory
for DIR in "$CONV_INPUT_DIR"/*; do
    for ANNOTATOR_PROMPT_PATH in "${ANNOTATOR_PROMPT_PATHS[@]}"; do
        bash /home/dimits/Documents/university/research/llm_moderation/experiments/scripts/annotation_execute_all.sh \
        --python_script_path "$PYTHON_SCRIPT_PATH" \
        --conv_input_dir "$DIR" \
        --prompt_path "$ANNOTATOR_PROMPT_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --model_path "$MODEL_PATH"
    done
done
