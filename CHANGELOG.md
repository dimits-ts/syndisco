# What's new

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