Overview
========

SynDisco is a Python library which creates, manages, and stores the logs of
synthetic ``discussions`` (discussions performed entirely by LLMs).

Each synthetic discussion is performed by ``actors``; actors can be
``user-agents`` (who simulate human users), ``moderators`` (who simulate
human chat moderators), and ``annotator-agents`` (who judge the discussions
after they have concluded).

    Example: A synthetic discussion takes place between Peter32 and Leo59
    (user-agents) and is monitored by Moderator1 (moderator). Later on, we
    instruct George12 and JohnFX to tell us how toxic each comment in the
    discussion is (annotator-agents).

Since social experiments are usually conducted at a large scale, SynDisco
manages discussions through ``experiments``. Each experiment is composed of
numerous discussions. Most of the variables in an experiment are randomized
to simulate real-world variation, while some are pinned in place by us.

    Example: We want to test whether the presence of a moderator impacts
    synthetic discussions. We create Experiment1 and Experiment2, where Exp1
    has a moderator and Exp2 does not. Both experiments will generate 100
    discussions using randomly selected users. In the end, we compare the
    toxicity between the discussions to resolve our hypothesis.

In general, each discussion goes through three phases:
``generation`` (according to the parameters of an experiment),
``execution``, and ``annotation``.

See how you can easily use these concepts programmatically in the
`Guides section <guides.md>`_.



