.. syndisco documentation master file, created by
   sphinx-quickstart on Tue Apr  1 16:18:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SynDisco's documentation!
====================================


.. image:: syndisco_logo.svg

Welcome to the official documentation of SynDisco! 
This project provides a lightweight framework for creating, managing, 
annotating, and analyzing synthetic discussions between Large Language Model 
(LLM) user-agents in online forums.
 
While simple, SynDisco offers multiple ways to randomize / customize 
discussions.
 
Features
========

- **Automated Experiment Generation**  
  SynDisco generates a randomized set of discussion templates. With only a
  handful of configurations, the researcher can run hundreds or thousands of
  unique experiments.

- **Synthetic Group Discussion Generation**  
  SynDisco accepts an arbitrarily large number of LLM user-agent profiles and
  possible Original Posts (OPs). Each experiment involves a random selection
  of these user-agents replying to a randomly selected OP. The researcher can
  determine how these participants behave, whether there is a moderator
  present, and even how the turn-taking is determined.

- **Synthetic Annotation Generation with multiple annotators**  
  The researcher can create multiple LLM annotator-agent profiles. Each of
  these annotators will process each generated discussion at the
  comment-level and annotate according to the provided instruction prompt,
  enabling an arbitrary selection of metrics to be used.

- **Native Transformers support**  
  The framework supports most Hugging Face Transformer models out of the box.
  Support for models managed by other libraries can be easily achieved by
  extending a single class.

- **Native logging and fault tolerance**  
  Since SynDisco may run for days on remote servers, it keeps detailed logs
  both on-screen and on-disk. Should any experiment fail, the next one will
  be loaded with no delay. Results are intermittently saved to disk, ensuring
  no data loss or corruption even in catastrophic errors.


Introduction
====================================
.. toctree::
   :maxdepth: 4

   overview
   installation
   guides
   syndisco


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
