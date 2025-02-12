# SynDisco: Automated experiment creation and execution using only LLM agents

A lightweight, simple and specialized framework used for creating, storing, annotating and analyzing synthetic discussions between LLM users in the context of online discussions.

This framework is designed for academic use, mainly for simulating Social Science experiments with multiple participants. It is finetuned for heavy server-side use and multi-day computations with limited resources. It has been tested on both simulated debates and online fora.

This repository only houses the source code for the framework. Input data, generated datasets, and analysis can be found in [this project](https://github.com/dimits-ts/synthetic_moderation_experiments).


## Features

#### Automated Experiment Generation

SynDisco generates a randomized set of social experiences each time. With only a handful of configurations, the researcher can run hundreds or thousands of experiments.

#### Synthetic Group Discussion Generation

SynDisco accepts an arbitrarily large number of LLM user-agent profiles and possible Original Posts (OPs). Each experiment involves a random selection of these user-agents replying to a randomly selected OP. The researcher can determine how these participants behave, whether there is a moderator present and even how the turn-taking is determined.

#### Synthetic Annotation Generation with multiple annotators

The researcher can create multiple LLM annotator-agent profiles. Each of these annotators will process each generated discussion at the comment-level, and annotate according to the provided instruction prompt, enabling an arbitrary selection of metrics to be used.

#### Native Transformers and llama.cpp support

The framework natively accepts any model loaded with the libraries above. Extending the framework to accept models loaded with other libraries can be trivially achieved by extending the base [Model class](src/sdl/backend/model.py) and by overriding the two methods. 

#### Native logging and fault tolerance

Since SynDisco is expected to possibly run for days at a time in remote servers it keeps detailed logs both on-screen and on-disk. Additionally, should any experiment fail, the next one will be loaded. Results are intermittently saved to the disk, ensuring no data loss or corruption on even catastrophic errors.


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

Currently the framework supports the `llama-cpp-python` and `transformers` libraries as a backend for loading and managing the underlying LLMs. Thus, any model supported by these libraries may be used. 

## Usage

* A YAML file containing the experiment configurations. [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/server_config.yml).

* Two `.txt` files for user and moderator instructions respectively (`user_instruction_path`, `moderator_instruction_path`). Examples for [user instructions](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/user_instructions/vanilla.txt) and [moderator instructions](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/mod_instructions/no_instructions.txt).

* A `.json` file containing general configurations for the conversation (`configs_path`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/other_configs/standard_multi_user.json).

* A directory containing `.txt` files, each containing a starting comment for the conversation (`topics_dir`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/topics/polarized_3.txt).

* A directory containing `.json` files representing the user personas (`persona_dir`). [Example file](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/data/generated_discussions_input/modular_configurations/personas/chill_2.json).

**This project is still in development. High-level documentation will soon be available**
