# SynDisco: Automated experiment creation and execution using only LLM agents

A lightweight, simple and specialized framework used for creating, storing, annotating and analyzing synthetic discussions between LLM users in the context of online discussions.

This framework is designed for academic use, mainly for simulating Social Science experiments with multiple participants. It is finetuned for heavy server-side use and multi-day computations with limited resources. It has been tested on both simulated debates and online fora.

> ⚠ **Warning: Active Development**  
> This project is currently in active development. The API is subject to change at any time without prior notice.  
> We recommend keeping an eye on updates and version releases if you're using this project in your applications.
> Any bug reports or feature requests are welcome at this stage in development.


## Features

#### Automated Experiment Generation

SynDisco generates a randomized set of social experiences each time. With only a handful of configurations, the researcher can run hundreds or thousands of experiments.

#### Synthetic Group Discussion Generation

SynDisco accepts an arbitrarily large number of LLM user-agent profiles and possible Original Posts (OPs). Each experiment involves a random selection of these user-agents replying to a randomly selected OP. The researcher can determine how these participants behave, whether there is a moderator present and even how the turn-taking is determined.

#### Synthetic Annotation Generation with multiple annotators

The researcher can create multiple LLM annotator-agent profiles. Each of these annotators will process each generated discussion at the comment-level, and annotate according to the provided instruction prompt, enabling an arbitrary selection of metrics to be used.

#### Native Transformers support

Extending the framework to accept models loaded with other libraries can be trivially achieved by extending the base [Model class](src/sdl/backend/model.py) and by overriding the two methods. 

#### Native logging and fault tolerance

Since SynDisco is expected to possibly run for days at a time in remote servers it keeps detailed logs both on-screen and on-disk. Additionally, should any experiment fail, the next one will be loaded. Results are intermittently saved to the disk, ensuring no data loss or corruption on even catastrophic errors.


## Project Structure

The project is structured as follows:

* `src/scripts/`: automation scripts for batch processing of experiments 
* `src/sdl/`: the *Synthetic Discussion Library*, containing the necessary modules for synthetic discussion creation and annotation
* `src/run.py`: main script handling generation, annotation and export to csv


## Requirements

The code is currently tested for Linux only, but should run on any platform. The environment can be installed using `pip install -r requirements.txt`.


## Usage

A sample file using the framework can be found [here](https://github.com/dimits-ts/synthetic_moderation_experiments/blob/master/run.py).

