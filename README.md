# Synthetic Discussion Framework (SDF)

Continuation of the [sister thesis project](https://github.com/dimits-ts/llm_moderation_research). A lightweight, simple and specialized framework used for creating, storing, annotating and analyzing
synthetic discussions between LLM users in the context of online discussions. Used as a proxy for experimentation with human users when
researching optimal LLM moderation techniques.

## Requirements

### Environment & Dependencies

The code is tested for Linux only. The platform-specific (Linux x86 / NVIDIA CUDA) conda environment used in this project can be found up-to-date [here](https://github.com/dimits-ts/conda_auto_backup/blob/master/llm.yml).

Run [`src/scripts/download_model.sh`](src/scripts/download_model.sh) in order to download the model used to run the framework in the correct directory (~5 GB of storage needed).

## Use

### Setting up configurations

The framework is intended to be used with modular input files, which are then combined in various combinations to generate the final conversation inputs.

Default configurations are already provided. To modify and add configurations, simply change/add files in the [`data/generated_discussion_input/modular_configurations`](data/generated_discussion_input/modular_configurations) directory.

To generate the final conversation inputs run the [`src/scripts/generate_conversation_inputs.sh`](src/scripts/generate_conv_configs_personalized.sh) script.

### Synthetic conversation creation

There are many ways with which to use the synthetic conversation framework:
1. (Preferred) Run [`src/scripts/conversation_personalized.sh`](src/scripts/conversation_personalized.sh), where `output_dir` is the directory of the final conversation inputs (see section above) 
1. Create a new python script leveraging the framework library found in the `sdl` module



## Structure

The project is structured as follows:

- `data`: SDF input configurations and output
- `src/models`: directory for local LLM instances
- `src/scripts`: automation scripts for batch processing of experiments and conversation input creation
- `src/sdl`: the Synthetic Discussion Library, containing the necessary modules for synthetic discussion creation and annotation

Notable files:
- [`src/sdf_create_conversations.py`](src/sdf_create_conversations.py): script automatically loading a conversation's parameters, executing the synthetic dialogue using a local LLM, and serializing the output
- [`src/sdf_create_annotations.py`](src/sdf_create_annotations.py): script loading a previously concluded conversation from serialized data, executing an annotation job using a local LLM, and serializing the output
- [`src/generate_conv_configs.py`](src/generate_conv_configs.py): a notebook containing notes on the experiments, implementation and design details, as well as example code for our framework

## Documentation

Since the project is still nascent and its API constantly shifts, there is no separate, stable documentation. However, we provide up-to-date documentation in the docstrings found in the python source files.