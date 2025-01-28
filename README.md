# SynDisco: Automated experiment creation and execution using only LLM agents

Continuation of the [sister thesis project](https://github.com/dimits-ts/llm_moderation_research). A lightweight, simple and specialized framework used for creating, storing, annotating and analyzing
synthetic discussions between LLM users in the context of online discussions.

This repository only houses the source code for the framework. Input data, generated datasets, and analysis can be found in [this project](https://github.com/dimits-ts/synthetic_moderation_experiments).

## Project Structure

The project is structured as follows:

* `tests/`: self-explanatory
* `src/scripts/`: automation scripts for batch processing of experiments 
* `src/sdl/`: the *Synthetic Discussion Library*, containing the necessary modules for synthetic discussion creation and annotation
* `src/run.py`: main script handling generation, annotation and export to csv

## Requirements

### Environment & Dependencies

The code is tested for Linux only, but can run on any platform. The platform-specific (Linux x86 / NVIDIA CUDA) conda environment used in this project can be found up-to-date [here](https://github.com/dimits-ts/conda_auto_backup/blob/master/llm.yml).

### Supported Models

Currently the framework only supports the `llama-cpp-python` and `transformers` libraries as a backend for loading and managing the underlying LLMs. Thus, any model supported by these libraries may be used. 

## Usage

* A YAML file containing the experiment configurations. [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/server_config.yml).

* Two `.txt` files for user and moderator instructions respectively (`user_instruction_path`, `moderator_instruction_path`). Examples for [user instructions](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/user_instructions/vanilla.txt) and [moderator instructions](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/mod_instructions/no_instructions.txt).

* A `.json` file containing general configurations for the conversation (`configs_path`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/other_configs/standard_multi_user.json).

* A directory containing `.txt` files, each containing a starting comment for the conversation (`topics_dir`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/topics/polarized_3.txt).

* A directory containing `.json` files representing the user personas (`persona_dir`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/personas/chill_2.json).

**This project is still in development. High-level documentation will soon be available**


                                    [--gpu_layers GPU_LAYERS]
```
