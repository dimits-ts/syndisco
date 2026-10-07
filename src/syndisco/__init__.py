"""
A lightweight framework for creating, managing, annotating, and analyzing
synthetic discussions between Large Language Model (LLM) user-agents.
"""

import importlib
import typing

__version__ = "2.3.0"


# Public names are imported lazily: ``import syndisco`` is cheap, and heavy
# dependencies (torch, transformers, openai) are only loaded the first time a
# class that needs them is accessed. This keeps lightweight entry points, such
# as the ``syndisco view`` command, fast to start.
__all__ = [
    "DiscussionExperiment",
    "AnnotationExperiment",
    "Actor",
    "Discussion",
    "Annotation",
    "Logs",
    "logging_setup",
    "BaseModel",
    "TransformersModel",
    "OpenAIModel",
    "TurnManager",
    "RespondTurnManager",
    "RandomTurnManager",
    "QueueTurnManager",
]

# public name -> submodule that defines it
_LAZY_ATTRS: dict[str, str] = {
    "DiscussionExperiment": "experiments",
    "AnnotationExperiment": "experiments",
    "Actor": "actors",
    "Discussion": "jobs",
    "Annotation": "jobs",
    "Logs": "jobs",
    "logging_setup": "logging",
    "BaseModel": "model",
    "TransformersModel": "model",
    "OpenAIModel": "model",
    "TurnManager": "turn_manager",
    "RespondTurnManager": "turn_manager",
    "RandomTurnManager": "turn_manager",
    "QueueTurnManager": "turn_manager",
}

if typing.TYPE_CHECKING:  # pragma: no cover
    from .experiments import DiscussionExperiment, AnnotationExperiment
    from .actors import Actor
    from .jobs import Discussion, Annotation, Logs
    from .logging import logging_setup
    from .model import TransformersModel, OpenAIModel, BaseModel
    from .turn_manager import (
        RespondTurnManager,
        QueueTurnManager,
        RandomTurnManager,
        TurnManager,
    )


def __getattr__(name: str) -> typing.Any:
    submodule = _LAZY_ATTRS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f".{submodule}", __name__), name)
    globals()[name] = value  # cache, so __getattr__ runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
