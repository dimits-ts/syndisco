.. syndisco documentation master file, created by
   sphinx-quickstart on Tue Apr  1 16:18:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SynDisco's documentation!
====================================


.. image:: logo.svg

Welcome to the official documentation of SynDisco! 
This project provides a lightweight framework for creating, managing, annotating, and analyzing synthetic discussions between Large Language Model (LLM) user-agents.
 
While simple, SynDisco offers multiple ways to randomize / customize discussions.
 
Features
========

- **Automated Experiment Generation**  
  SynDisco generates a randomized set of discussion templates. With only a
  handful of configurations, the library can execute hundreds or thousands of unique experiments.

- **Synthetic Group Discussion Generation**  
  SynDisco accepts an arbitrarily large number of LLM user-agent profiles and possible seed comments (parts of real-life discussions). Each experiment involves a random selection of these user-agents replying to a randomly selected series of comments.

- **Synthetic Annotation Generation with multiple annotators**  
  The library can create multiple LLM annotator-agent profiles. Each of
  these annotators will process each generated discussion at the
  comment-level and annotate according to the provided instruction prompt,
  enabling an arbitrary selection of metrics to be used.

- **Support for most models**  
  Our library supports most Hugging Face Transformer models out of the box for local execution, as well as all OpenAI models, and (soon!) Gemini.Support for models managed by other libraries can be easily achieved by extending a single class.

- **Native logging and fault tolerance**  
  Since SynDisco may run for days on remote servers, it keeps detailed logs both on-screen and on-disk. Should any experiment fail, the next one will be loaded with no delay. Results are intermittently saved to disk, ensuring no data loss or corruption even in catastrophic errors.


Introduction
====================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   guides
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
