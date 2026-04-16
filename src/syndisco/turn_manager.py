"""
Module handling the turn order of LLM participants in discussions.
"""

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

import abc
import random
import typing
from collections.abc import Iterable
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

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @abc.abstractmethod
    def _next_impl(self) -> Actor:
        raise NotImplementedError("Abstract method called")


class RoundRobin(TurnManager):
    """
    A simple turn manager which gives priority to the next user in the queue.
    """

    def __init__(self, actors: Iterable[Actor] | None = None):
        super().__init__(actors)
        self.curr_turn = -1

    def _next_impl(self) -> Actor:
        self.curr_turn += 1
        new_speaker_index = self.curr_turn % len(self._actors)
        return self._actors[new_speaker_index]


class RandomWeighted(TurnManager):
    """
    Enable a participant to reply with a set probability, else randomly select
    another participant.
    """

    def __init__(
        self,
        actors: Iterable[Actor] | None = None,
        p_respond: float = 0,
    ):
        super().__init__(actors)

        assert (
            0 <= p_respond <= 1
        ), f"p_respond must be between 0 and 1, but is {p_respond}"

        self._chance_to_respond = p_respond

        # Track history
        self._last_speaker: Actor | None = None
        self._second_to_last_speaker: Actor | None = None

    @property
    def chance_to_respond(self) -> float:
        """
        The chance that the second-to-last speaker will respond to the
        last speaker. Between 0 and 1.

        :return: The chance of responding.
        :rtype: float
        """
        return self._chance_to_respond

    @chance_to_respond.setter
    def chance_to_respond(self, p_respond: float) -> None:
        assert (
            0 <= p_respond <= 1
        ), f"p_respond must be between 0 and 1, but is {p_respond}"
        self._chance_to_respond = p_respond

    def _next_impl(self) -> Actor:
        # First turn: no history yet, pick random
        if self._last_speaker is None:
            next_speaker = self._random_actor()
        elif self._second_to_last_speaker is None:
            next_speaker = self._random_actor(exclude=self._last_speaker)
        else:
            if self._should_repeat_last_speaker():
                next_speaker = self._second_to_last_speaker
            else:
                next_speaker = self._random_actor(exclude=self._last_speaker)

        # Update history
        self._second_to_last_speaker = self._last_speaker
        self._last_speaker = next_speaker

        return next_speaker

    def _should_repeat_last_speaker(self) -> bool:
        """Return True if the last speaker should respond again."""
        return random.random() < self.chance_to_respond

    def _random_actor(self, exclude: Actor | None = None) -> Actor:
        """Select a random actor, optionally excluding one."""
        if exclude is None:
            return random.choice(self._actors)

        candidates = [actor for actor in self._actors if actor != exclude]

        # Fallback: if only one actor exists
        if not candidates:
            return exclude

        return random.choice(candidates)
