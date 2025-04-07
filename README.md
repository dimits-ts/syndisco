# SynDisco: Automated experiment creation and execution using only LLM agents

A lightweight, simple and specialized framework used for creating, storing, annotating and analyzing synthetic discussions between Large Language Model (LLM) user-agents in the context of online discussions.

This framework is designed for academic use, mainly for simulating Social Science experiments with multiple participants. It is finetuned for heavy server-side use and multi-day computations with limited resources. It has been tested on both simulated debates and online fora.

> ⚠ **Warning: Active Development**  
> This project is currently in active development. The API is subject to change at any time without prior notice.  
> We recommend keeping an eye on updates and version releases if you're using this project in your applications.
> Any bug reports or feature requests are welcome!.


## Usage

Have a look at the [online documentation](https://dimits-ts.github.io/syndisco/) for high-level descriptions, API documentation and tutorials.


## Features

#### Automated Experiment Generation

SynDisco generates a randomized set of discussion templates. With only a handful of configurations, the researcher can run hundreds or thousands of unique experiments.

#### Synthetic Group Discussion Generation

SynDisco accepts an arbitrarily large number of LLM user-agent profiles and possible Original Posts (OPs). Each experiment involves a random selection of these user-agents replying to a randomly selected OP. The researcher can determine how these participants behave, whether there is a moderator present and even how the turn-taking is determined.

#### Synthetic Annotation Generation with multiple annotators

The researcher can create multiple LLM annotator-agent profiles. Each of these annotators will process each generated discussion at the comment-level, and annotate according to the provided instruction prompt, enabling an arbitrary selection of metrics to be used.

#### Native Transformers support

The framework supports most Hugging Face Transformer models out-of-the-box. Support for models managed by other libraries can be easily achieved by extending a single class. 

#### Native logging and fault tolerance

Since SynDisco is expected to possibly run for days at a time in remote servers, it keeps detailed logs both on-screen and on-disk. Should any experiment fail, the next one will be loaded with no intermittent delays. Results are intermittently saved to the disk, ensuring no data loss or corruption on even catastrophic errors.


## Installation

You can download the package from PIP:

```bash
pip install syndisco
```

Or build from source:
```bash
git clone https://github.com/dimits-ts/syndisco.git
pip install -r requirements.txt
pip install .
```

If you want to contribute to the project, or modify the library's code you may use:
```bash
git clone https://github.com/dimits-ts/syndisco.git
pip install -r requirements.dev.txt
pip install -e .
```