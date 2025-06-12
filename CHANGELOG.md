# What's new

## 2.0.0 (12/06/2025)
Note: Patch 2.0.0 introduces **breaking** cnanges in most of Syndisco's APIs. While the package is not stable yet, most of these changes were deemed crucial both for development, and to correct severe design oversights.

### Major changes
- Normalize module naming
    - All modules are now top-level, below the syndisco master module
    - Certain modules have been merged together to make the API more cohesive
- Remove unsupported functions
    - A part of the codebase proved to serve extremely niche use cases

### Additions
- Added progress bars to both jobs and experiments
- Added JSON support for all experiment, actor, and persona configurations
    - Besides helping development, this allows easy and uniform access across all logs and exported datasets
- Logs now create new files when the day changes

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