"""
SynDisco: Automated experiment creation and execution using only LLM agents
Copyright (C) 2025 Dimitris Tsirmpas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at tsirbasdim@gmail.com
"""

import abc
import random
import warnings
import typing
from collections.abc import Iterable


class TurnManager(Iterable):
    """
    A class that handles which handles turns between users.
    """

    def __init__(self, names: Iterable[str] | None = None):
        """
        Construct a new TurnManager.

        :param names: The usernames of the participants.
            Can be left null if names are to be decided
            after this object's creation.
        :type config: dict[str, float], optional
        """
        self.names = names

    @typing.final
    def set_names(self, names: Iterable[str]) -> None:
        """
        Initialize the manager by providing the names of the users.

        :param names: the usernames of the participants
        :type names: list[str]
        """
        self.names = names

    @typing.final
    def next(self) -> str:
        """
        Get the username of the next speaker.

        :raises ValueError: if no names have been provided from the
            constructor, or from the TurnManager.set() method
        :return: the next speaker's username
        :rtype: str
        """
        if self.names is None:
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
    def _next_impl(self) -> str:
        raise NotImplementedError("Abstract method called")


class RoundRobbin(TurnManager):
    """
    A simple turn manager which gives priority to the next user in the queue.
    """

    def __init__(self, names: Iterable[str] | None = None):
        super().__init__(names)
        self.curr_turn = -1

    def _next_impl(self) -> str:
        self.curr_turn += 1
        new_speaker_index = self.curr_turn % len(self.names)
        return self.names[new_speaker_index]


class RandomWeighted(TurnManager):
    """
    Enable a participant to reply with a set probability, else randomly select
    another participant.
    """

    DEFAULT_RESPOND_PROBABILITY = 0.5

    def __init__(
        self, p_respond: float = -1, names: Iterable[str] | None = None
    ):
        super().__init__(names)

        if p_respond == -1:
            warnings.warn(
                "Warning: No p_respond set in RandomWeighted instance, "
                + f"defaulting to {RandomWeighted.DEFAULT_RESPOND_PROBABILITY}"
            )
            self.chance_to_respond = RandomWeighted.DEFAULT_RESPOND_PROBABILITY
        else:
            self.chance_to_respond = p_respond
            assert (
                0 < self.chance_to_respond < 1
            ), f"p_respond must be between 0 and 1, but is {p_respond}"

        self.second_to_last_speaker = None
        self.last_speaker = None

    def _next_impl(self) -> str:
        # If first time asking for a speaker, return random speaker
        if self.second_to_last_speaker is None:
            next_speaker = self._select_other_random_speaker()
            self.last_speaker = next_speaker
            return next_speaker

        # Check if the last speaker will respond based on weighted coin flip
        if self._weighted_coin_flip():
            next_speaker = self.last_speaker
        else:
            next_speaker = self._select_other_random_speaker()

        # Update the speaker history
        self.second_to_last_speaker = self.last_speaker
        self.last_speaker = next_speaker

        assert next_speaker is not None
        return next_speaker

    def _weighted_coin_flip(self) -> bool:
        return self.chance_to_respond > random.uniform(0, 1)

    def _select_other_random_speaker(self) -> str:
        other_usernames = [
            username
            for username in self.names
            if username != self.last_speaker
        ]
        return random.choice(other_usernames)
