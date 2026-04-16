# What's new

## 2.1.1 (16/04/2026)

### Features
- Common classes and functions can now be directly called from the top-level module
    - E.g., previously, users had to import `syndisco.actors.Actor` instead of `syndisco.Actor`.
- All TurnManagers now directly manage actors (previously used strings for actor names)
- The Actor API has been overhauled
    - There is now a distinction between model name and actor name
    - The `describe()` method is now split between returning system and user prompts
    - The `ActorType` enum has been replaced by a constructor boolean for whether the actor is an annotator
- The `Annotation` jobs now accept `Logs` objects instead of reading files from directories
- The `TransformerModel` class now allows kwargs for model, tokenizer, and generation
- Replaced the redundant `Persona` class with python dictionaries
- Removed redundant timing logs
- Updated documentation and guides

### Development
- Re-added test suite
- Refactored old and clunky code
- Removed redundant modules such as syndisco._file_util
- Improved internal documentation

### Fixes
- Added correct equality comparisons to the `Logs` class using the `__eq__` operator instead of hashed ids.
- Abstract classes are now included in the public documentation
- Improved error handling for OpenAI models 
- Fixed RandomWeighted logic
- Fixed `Annotator` logs including the system instead of user prompts 


## 2.1.0 (03/04/2026)
Version 2.1.0 simplifies the API even more. The library is exclusively focused on 
general discussion generation (instead of facilitative discussions), 
and extraneous, easy-to-break, and hard-to-maintain functions have been retired.


### Features
- The `Discussion`, `Annotation`, `DiscussionExperiment`, and `AnnotationExperiment` classes
no longer natively support facilitator (moderator) agents by default.
    - Instead, moderator agents can be modelled as regular agents with special 
    instruction prompts, with a slight modification of the turn-taking function
    to pass them priority after every comment.

- Data import and export is no longer handled by the `Discussion` and `Annotation` 
classes. Instead, all persistent state is managed by the new `Logs` class.
    - This distinction merges the unreliable and unintuitive ways of loading 
    serialized outputs into a single, simple API for both classes.
    - The `Discussion` and `Annotation` classes are now exclusively used for
    runtime execution, which should simplify the API for new users.

- The `Discussion` class now allows for step-by-step execution.
    - Specifically, it's now an iterator that can be called in the same way as 
    a `list` or `range` would.
    - This change allows for interactive approaches or the implementation of
    a REST API in the future.

- The `postprocessing` module has been retired.
    - Since all logs are serialized in a standard JSON format, it makes little 
    sense also supporting CSV exports.
    - This module historically had the most problematic support, with frequent 
    bugs and schema breaks. It was also very hard to maintain, since any implicit
    change in any dataclass broke the code.
    - Instead of rewriting the entire module, we are retiring it and instead
    pull our efforts into making the JSON serializer more intuitive.


### Fixes

- Common classes have now (finally) been moved up to the top module.
    - This change allows users to call `syndisco.Discussion` instead of
    `syndisco.jobs.Discussion`, which had quickly become annoying.

- Persona prompts now only include specified demographic dimensions.
    - Up until now, leaving fields blank would include these fields in the prompt.
    - This minor change will make some prompts shorter, which should make a difference
    when using local models at scale.

- Improved documentation website and expand tutorials.


## 2.0.8 (17/3/2026)
- Reworked discussion and annotation exports to be more robust, intuitive and maintainable.
- Transformer models are now loaded in eval mode and run on inference mode.
    - This should lead to some improvements w.r.t. memory and computational requirements, but don't expect anything drastic.
    - I am still amazed that these were not the defaults for the huggingface models. The documentation about this is sorely lacking.
    - Thanks to my colleague T.P. for pointing this inefficiency out!

## 2.0.7 (23/1/2025)
- Fixed long-standing bugs with annotation export
- Updated guides
- Fixed dependency issues from 2.0.6

## 2.0.6 (6/12/2025)
- Added OpenAI support

## 2.0.5 (26/11/2025)

### Features 
- Improved documentation

### Fixes
- Removed conv_variant column from preprocessing, since it was too use-case specific
- Sphinx uses conventional configuration
- Docs are now built using gh-actions instead of pushing all HTML pages to Github


## 2.0.4+ (26/11/2025)

### Features 
- Added logo :)

### Fixes

- Fixed postprocessing bug not properly parsing JSON files
- Fixed postprocessing discussions not handling moderator properly
- Fixed inconsistent hashing between program restarts
- Fixed outdated documentation in some parts

## 2.0.3 (21/11/2025)

### Features
- The logger now asks gently before spamming stdout
    - Removed redundant logging messages in experiments
    - Downgraded timing information to DEBUG stream
- Remove word stop-list
    - It's almost 2026, and LLMs are now much more stable in their output

### Fixes
- The `WeightedRandom` turn-taking algorithm now accepts {0,1} values (corresponding to "never/always select the previous speaker").
- Fix issues between internal Actor/Model modules when using models other than LLaMa


## 2.0.2 (17/11/2025)

### Features
- Multiple seed opinions can be given for each synthetic discussion
- Usernames do not have to be random when giving seed opinions via the Experiments interface

### Fixes
- The documentation page actually updates now
- Fixed bug that prevented persona loading from json files
- Fixed persistent issues with packaging discussion files into csvs

## 2.0.1 (13/06/2025)

### Features
- Replace conda environments with pypi requirements

### Fixes
- Fix progress bars not working properly in the experiment level


## 2.0.0 (12/06/2025)
Note: Patch 2.0.0 introduces **breaking** cnanges in most of Syndisco's APIs. While the package is not stable yet, most of these changes were deemed crucial both for development, and to correct design oversights.

### Features
- Added progress bars to both jobs and experiments
- Added JSON support for all experiment, actor, and persona configurations
    - Besides helping development, this allows easy and uniform access across all logs and exported datasets
- Logs now create new files when the day changes

### Changes
- Normalize module naming
    - All modules are now top-level, below the syndisco master module
    - Certain modules have been merged together to make the API more cohesive
- Remove unsupported functions
    - A part of the codebase proved to serve extremely niche use cases

### Fixes
- Fix module-level documentation being replaced with licence information
- Fix Experiments only allowing verbose generation
- Documentation fixes
- Various bug fixes



## 1.0.2 (11/04/2025)
- Rename Round Robin turn manager
- Fix bug where Round Robin would crash when used by a DiscussionExperiment
- Updated dev environments

## 1.0.1 (09/04/2025)

- Fix issue with online documentation not showing modules
- Include pytorch as default transformers engine
- Update conda environments