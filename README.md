# SynDisco: Automated experiment creation and execution using only LLM agents

![Syndisco Logo](./docs/source/logo.svg)

A lightweight, simple and specialized library used for creating, storing, annotating and analyzing synthetic discussions between Large Language Model (LLM) user-agents. 

Unlike other libraries attempting to streamline LLM interactions, syndisco:
- Does not load any VRAM modules except for the underlying LLM
- Does not run any prompts on the LLM other than the prompt to speak in the discussion
- Has a very simple and easy-to-learn API
- Allows the use of local LLMs (although support for proprietary models is being added)
- Is completely free and open source
- Is finetuned for heavy server-side use and multi-day computations with limited resources.

## Description and Usage

Have a look at the [online documentation](https://dimits-ts.github.io/syndisco/) for high-level descriptions, API documentation, and tutorials.


## Installation

You can download the package from PIP:

```bash
pip install syndisco
```

Or build from source:
```bash
git clone https://github.com/dimits-ts/syndisco.git
cd syndisco
pip install .
```

If you want to contribute to the project, or modify the library's code you may use:
```bash
git clone https://github.com/dimits-ts/syndisco.git
cd syndisco
pip install -e .[dev]
```
