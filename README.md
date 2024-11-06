# Synthetic Discussion Framework (SDF)

Continuation of the [sister thesis project](https://github.com/dimits-ts/llm_moderation_research). A lightweight, simple and specialized framework used for creating, storing, annotating and analyzing
synthetic discussions between LLM users in the context of online discussions.

This repository only houses the source code for the framework. Input data, generated datasets, and analysis can be found in [this project](https://github.com/dimits-ts/synthetic_moderation_experiments).

## Project Structure

The project is structured as follows:

* `tests/`: self-explanatory
* `src/scripts/`: automation scripts for batch processing of experiments 
* `src/sdl/`: the *Synthetic Discussion Library*, containing the necessary modules for synthetic discussion creation and annotation
* `src/generate_conv_configs.py`: Sets up synthetic experiment
* `src/generate_annotation_configs.py`: Sets up synthetic annotation job
* `src/generate_conversations.py`: Runs synthetic conversation experiment
* `src/generate_annotations.py`: Runs synthetic annotation job on synthetic conversations

## Requirements

### Environment & Dependencies

The code is tested for Linux only. The platform-specific (Linux x86 / NVIDIA CUDA) conda environment used in this project can be found up-to-date [here](https://github.com/dimits-ts/conda_auto_backup/blob/master/llm.yml).

### Supported Models

Currently the framework only supports the `llama-cpp-python` library as a backend for loading and managing the underlying LLMs. Thus, any model supported by the `llama-cpp-python` library may be used. A complete list of models can be found [here](https://github.com/ggerganov/llama.cpp).

## Usage

### Generating the experiment files

The framework is intended to be used with modular input files, which are then combined in various combinations to generate the final inputs for the conversation/annotation jobs. This is a convenient and intuitive way to set up diverse experiments, but requires extra steps to initially set up.

For default configurations visit [this project](https://github.com/dimits-ts/synthetic_moderation_experiments). It is recommended to use the default configurations as a starting point, and later modify them to suit your needs.

To generate custom experiments, use the [`generate_conv_configs.py`](src/generate_conv_configs.py) and [`generate_annotation_configs.py`](src/generate_annotation_configs.py) scripts respectively. These scripts assume modular inputs of a specific format.

#### Generating conversational experiment jobs

```bash
usage: generate_conv_configs.py [-h] --output_dir OUTPUT_DIR 
                                        --persona_dir PERSONA_DIR 
                                        --topics_dir TOPICS_DIR
                                        --configs_path CONFIGS_PATH 
                                        --user_instruction_path USER_INSTRUCTION_PATH
                                        --mod_instruction_path MOD_INSTRUCTION_PATH
                                        [--num_generated_files NUM_GENERATED_FILES] 
                                        [--num_users NUM_USERS]
                                        [--include_mod | --no-include_mod]
```

The script needs:

* Two `.txt` files for user and moderator instructions respectively (`user_instruction_path`, `moderator_instruction_path`). Examples for [user instructions](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/user_instructions/vanilla.txt) and [moderator instructions](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/mod_instructions/no_instructions.txt).

* A `.json` file containing general configurations for the conversation (`configs_path`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/other_configs/standard_multi_user.json).

* A directory containing `.txt` files, each containing a starting comment for the conversation (`topics_dir`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/topics/polarized_3.txt).

* A directory containing `.json` files representing the user personas (`persona_dir`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/personas/chill_2.json).

#### Generating annotation jobs

```bash
usage: generate_annotation_configs.py [-h] --output_dir OUTPUT_DIR 
                                            --persona_dir PERSONA_DIR
                                            --instruction_path INSTRUCTION_PATH 
                                            [--history_ctx_len HISTORY_CTX_LEN]
                                            [--include_mod_comments | --no-include_mod_comments]
```

The script needs:

<!-- TODO: update links -->
* A `.txt` file for annotator instructions (`instruction_path`). [Example file](aaaaaaaaaaa)

* A directory containing `.json` files representing the LLM annotator-agents personas (`persona_dir`). These have the exact same schema as the LLM user-agent personas above. [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/personas/chill_2.json).

### Synthetic Dataset Generation

#### Synthetic Discussion Generation

1. (Preferred) Run [src/scripts/conversation_execute_all.sh](src/scripts/conversation_execute_all.sh)

1. Use the [src/generate_conversations.py](src/generate_conversations.py) python script to generate a single conversation

1. Create a new python script leveraging the framework library found in the `sdl` module.

Bash script usage:

```bash
usage: scripts/conversation_execute_all.sh --python_script_path <python script path> --input_dir <input_directory> --output_dir <output_directory> --model_path <model_file_path>

```

Python script usage:

```bash
usage: generate_conversations.py [-h] --input_file INPUT_FILE 
                                        --output_dir OUTPUT_DIR 
                                        --model_path MODEL_PATH
                                        [--max_tokens MAX_TOKENS] 
                                        [--ctx_width_tokens CTX_WIDTH_TOKENS]
                                        [--random_seed RANDOM_SEED] [--inference_threads INFERENCE_THREADS]
                                        [--gpu_layers GPU_LAYERS]

```

### Synthetic annotation generation

1. (Preferred) Run [src/scripts/annotation_execute_all.sh](src/scripts/annotation_execute_all.sh) to annotate all conversations in a directory.

1. Use the [src/generate_annotations.py](src/generate_annotations.py) python script to generate annotations for a single conversation.

1. Create a new python script leveraging the framework library found in the `sdl` module.

Bash script usage:

```bash
usage: scripts/annotation_execute_all.sh --python_script_path <python script path> --conv_input_dir <input_directory> --prompt_path <input_path> --output_dir <output_directory> --model_path <model_file_path>

```

Python script usage:

```bash
usage: generate_annotations.py [-h] --prompt_input_path PROMPT_INPUT_PATH 
                                    --conv_path CONV_PATH 
                                    --output_dir OUTPUT_DIR 
                                    --model_path MODEL_PATH
                                    [--max_tokens MAX_TOKENS]
                                    [--ctx_width_tokens CTX_WIDTH_TOKENS] 
                                    [--random_seed RANDOM_SEED]
                                    [--inference_threads INFERENCE_THREADS]
                                    [--gpu_layers GPU_LAYERS]
```
