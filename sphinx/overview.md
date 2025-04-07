## Overview

SynDisco is a Python library which creates, manages and stores the logs of synthetic `discussions` (discussions performed entirely by LLMs). 

Each synthetic discussion is performed by `actors`; actors can be `user-agents` (who simulate human users), `moderators` (who simulate human chat moderators) and `annotator-agents` (who judge the discussions after they have concluded).

> Example: A synthetic discussion takes place between Peter32 and Leo59 (user-agents) and is monitored by Moderator1 (moderator). Later on, we instruct George12 and JohnFX to tell us how toxic each comment in the discussion is (annotator-agents). 

Since social experiments are usually conducted at a large scale, SynDisco manages discussions through `experiments`. Each experiment is composed of numerous discussions. Most of the variables in an experiment are randomized to simulate real-world variation, while some are pinned in place by us.

> Example: We want to test whether the presence of a moderator impacts synthetic discussions. We create Experiment1 and Experiment2, where Exp1 has a moderator and Exp2 does not. Both experiments will generate 100 discussions using randomly selected users. In the end, we compare the toxicity between the discussions to resolve our hypothesis.

In general, each discussion goes through three phases: `generation` (according to the parameters of an experiment), `execution`, and `annotation`.

See how you can easily use these concepts programmatically in the [Guides section](guides.md).

## Features

- **Automated Experiment Generation**  
  SynDisco generates a randomized set of discussion templates. With only a handful of configurations, the researcher can run hundreds or thousands of unique experiments.

- **Synthetic Group Discussion Generation**  
  SynDisco accepts an arbitrarily large number of LLM user-agent profiles and possible Original Posts (OPs). Each experiment involves a random selection of these user-agents replying to a randomly selected OP. The researcher can determine how these participants behave, whether there is a moderator present and even how the turn-taking is determined.

- **Synthetic Annotation Generation with multiple annotators**  
  The researcher can create multiple LLM annotator-agent profiles. Each of these annotators will process each generated discussion at the comment-level, and annotate according to the provided instruction prompt, enabling an arbitrary selection of metrics to be used.

- **Native Transformers support**  
  The framework supports most Hugging Face Transformer models out-of-the-box. Support for models managed by other libraries can be easily achieved by extending a single class.

- **Native logging and fault tolerance**  
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
