# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2025 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr
"""
Module handling the turn order of LLM participants in discussions.
"""

import abc
import copy
import typing
import warnings
from collections.abc import Iterable

import numpy as np

from .actors import Actor


class TurnManager(Iterable, abc.ABC):
    """
    An abstract class specifying the selection of the next speaker in a
    :class:`Discussion`.
    """

    def __init__(self, actors: Iterable[Actor] | None = None):
        """
        Construct a new TurnManager.

        :param actors: The participants.
            Can be left null if the participants are to be decided
            after this object's creation.
        :type actors: Iterable[Actor]
        """
        if actors is None:
            self._actors = []
        else:
            self._actors = list(actors)

    @typing.final
    def set_actors(self, actors: typing.Sequence[Actor]) -> None:
        """
        Initialize the manager by providing the names of the users.

        :param names: The participants.
        :type names: Sequence[Actor]
        """
        self._actors = list(actors)

    @typing.final
    def next(self) -> Actor:
        """
        Get the username of the next speaker.

        :raises ValueError:
            if no names have been provided from the
            constructor, or from the :meth:`set_actors()` method
        :return: the next speaker's username
        :rtype: Actor
        """
        if self._actors == []:
            raise ValueError(
                "No usernames have been provided for the turn manager. "
                "Use self.initialize_names()"
            )
        return self._next_impl()

    def make_instance(self) -> typing.Self:
        """
        Return a fresh copy of this manager with static configuration
        preserved and per-discussion state reset.

        Called once per discussion by :class:`DiscussionExperiment`.
        Subclasses with additional stateful attributes should override
        this method and reset those attributes on the returned instance.
        """
        instance = copy.copy(self)
        instance._actors = []
        # subclasses with extra mutable state should override this
        return instance

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @abc.abstractmethod
    def _next_impl(self) -> Actor:
        raise NotImplementedError("Abstract method called")


class QueueTurnManager(TurnManager):
    """
    A simple turn manager which gives priority to the next user in the queue.
    """

    def __init__(
        self,
        actors: Iterable[Actor] | None = None,
        randomize_first_speaker: bool = False,
        random_state: np.random.Generator | None = None,
    ):
        super().__init__(actors)

        self._randomize_first_speaker = randomize_first_speaker
        self._curr_turn = None

        self._rng = random_state or np.random.default_rng()

    def _next_impl(self):
        if self._curr_turn is None:
            # the second condition exists just show the linter shuts up
            if self._randomize_first_speaker:
                start_index = self._rng.integers(-1, len(self._actors))
            else:
                start_index = -1

            self._curr_turn = start_index

        self._curr_turn += 1
        return self._actors[self._curr_turn % len(self._actors)]

    def make_instance(self):
        instance = super().make_instance()
        instance._curr_turn = None
        return instance


class RespondTurnManager(TurnManager):
    """
    Enable a participant to reply with a set probability, else randomly select
    another participant.
    """

    def __init__(
        self,
        actors: Iterable[Actor] | None = None,
        p_respond: float = 0.5,
        random_state: np.random.Generator | None = None,
    ):
        super().__init__(actors)

        # Keep assertions for robust input validation
        assert (
            0 <= p_respond <= 1
        ), f"p_respond must be between 0 and 1, but is {p_respond}"

        if p_respond == 0:
            warnings.warn(
                """
                p_respond has been set to 0, which disables responding
                altogether. In that case, it may be better to use the
                RandomTurnManager class instead.
                """
            )

        self._chance_to_respond = p_respond

        self._rng = random_state or np.random.default_rng()
        self._last_speaker: Actor | None = None
        self._second_to_last_speaker: Actor | None = None

    def make_instance(self) -> typing.Self:
        instance = super().make_instance()
        instance._last_speaker = None
        instance._second_to_last_speaker = None
        # We rely on copy.copy() to handle the copying of the generator
        return instance

    @property
    def chance_to_respond(self) -> float:
        return self._chance_to_respond

    @chance_to_respond.setter
    def chance_to_respond(self, p_respond: float) -> None:
        assert (
            0 <= p_respond <= 1
        ), f"p_respond must be between 0 and 1, but is {p_respond}"
        self._chance_to_respond = p_respond

    def _next_impl(self) -> Actor:
        next_speaker = self._choose_next_speaker()
        self._second_to_last_speaker = self._last_speaker
        self._last_speaker = next_speaker
        return next_speaker

    def _choose_next_speaker(self) -> Actor:
        if self._last_speaker is None:
            return self._random_actor()

        if self._second_to_last_speaker is None:
            return self._random_actor(exclude=self._last_speaker)

        if self._should_repeat_last_speaker():
            return self._second_to_last_speaker

        return self._random_actor(exclude=self._last_speaker)

    def _should_repeat_last_speaker(self) -> bool:
        """Return True if the last speaker should respond again."""
        return self._rng.random() < self.chance_to_respond

    def _random_actor(self, exclude: Actor | None = None) -> Actor:
        """Select a random actor, optionally excluding one."""
        if exclude is None:
            num_actors = len(self._actors)
            random_index = self._rng.integers(low=0, high=num_actors)
            return self._actors[random_index]

        candidates = [actor for actor in self._actors if actor != exclude]

        if not candidates:
            return exclude

        num_candidates = len(candidates)
        random_index = self._rng.integers(low=0, high=num_candidates)
        return candidates[random_index]


class RandomTurnManager(RespondTurnManager):
    """
    Randomly chooses the next participant, excluding the last speaker.
    Functionally identical to :class:`RespondTurnManager` with ``p_respond=0``.
    """

    def __init__(
        self,
        actors: Iterable[Actor] | None = None,
        random_seed: int | None = None,
        random_state: np.random.Generator | None = None,
    ):
        super().__init__(actors=actors, p_respond=0, random_state=random_state)
