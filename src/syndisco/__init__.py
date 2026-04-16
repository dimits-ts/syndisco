"""
A lightweight framework for creating, managing, annotating, and analyzing
synthetic discussions between Large Language Model (LLM) user-agents.
"""
from .experiments import DiscussionExperiment, AnnotationExperiment
from .actors import Actor
from .jobs import Discussion, Annotation, Logs
from .logging import logging_setup
from .model import TransformersModel, OpenAIModel, BaseModel
from .turn_manager import RandomWeighted, RoundRobin, TurnManager

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
    "RandomWeighted",
    "RoundRobin"
]
