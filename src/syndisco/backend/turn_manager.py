
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


"""
Module deciding which Actor will speak next.
"""
import abc
import itertools
import random
import warnings
import typing


class TurnManager(abc.ABC):
    """
    A class that handles which of a list of users gets to speak in the next dialogue turn.
    """

    def __init__(self):
        """
        Construct a new TurnManager.

        :param config: a dictionary of other configurations, defaults to {}
        :type config: dict[str, float], optional
        """
        self.is_initialized = False

    @typing.final
    def initialize_names(self, usernames: list[str]) -> None:
        """
        Initialize the manager by providing the names of the users.

        :param usernames: the usernames of the participants
        :type usernames: list[str]
        """
        self.is_initialized = True
        self._initialize_names_impl(usernames)

    @typing.final
    def next(self) -> str:
        """
        Get the username of the next speaker.

        :return: the next speaker's username
        :rtype: str
        """
        if not self.is_initialized:
            raise ValueError(
                "No usernames have been provided for the turn manager. Use self.initialize_names()"
            )
        return self._next_impl()

    @abc.abstractmethod
    def _initialize_names_impl(self, usernames: list[str]) -> None:
        raise NotImplementedError("Abstract method called")

    @abc.abstractmethod
    def _next_impl(self) -> str:
        raise NotImplementedError("Abstract method called")


class RoundRobbin(TurnManager):
    """
    A simple turn manager which gives priority to the next user in the queue.
    """

    def __init__(self):
        super().__init__()
        self.username_loop = itertools.cycle([])
        self.curr_turn = 0

    def _initialize_names_impl(self, usernames: list[str]):
        self.username_loop = itertools.cycle(usernames)

    def _next_impl(self) -> str:
        return next(self.username_loop)


class RandomWeighted(TurnManager):
    """
    Enable a participant to reply with a set probability, else randomly select other participant.
    """

    DEFAULT_RESPOND_PROBABILITY = 0.5

    def __init__(self, config: dict[str, float] | None = None):
        super().__init__()
        config = config if config is not None else {}

        if config.get("respond_probability") is None:
            warnings.warn(
                "Warning: No respond_probability set in RandomWeighted TurnManager instance, "
                + f"defaulting to {RandomWeighted.DEFAULT_RESPOND_PROBABILITY}"
            )
            self.chance_to_respond = RandomWeighted.DEFAULT_RESPOND_PROBABILITY
        else:
            self.chance_to_respond = config["respond_probability"]
            assert 0 < self.chance_to_respond < 1

        self.second_to_last_speaker = None
        self.last_speaker = None
        self.usernames = {}

    def _initialize_names_impl(self, usernames: list[str]):
        self.usernames = set(usernames)

    def _next_impl(self) -> str:
        # If first time asking for a speaker, return random speaker
        if self.second_to_last_speaker is None:
            next_speaker = self._select_other_random_speaker()
            self.last_speaker = next_speaker
            return next_speaker

        # Check if the last speaker will respond based on the weighted coin flip
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
            username for username in self.usernames if username != self.last_speaker
        ]
        return random.choice(other_usernames)


def turn_manager_factory(
    turn_manager_type: str, config: dict[str, float] | None = None
) -> TurnManager:
    """
    A factory which returns a instansiated TurnManager of the type specified by a string.

    :param turn_manager_type: the string specifying the concrete TurnManager class.
    Can be of one of "round_robbin", "random_weighted"
    :type turn_manager_type: TurnManager
    :param usernames: a list of all usernames of each participant in the conversation
    :type usernames: list[str]
    :raises ValueError: if turn_manager_type does not match any classes
    :return: the instansiated TurnManager of the specified type
    :rtype: TurnManager
    """
    turn_manager_type = turn_manager_type.lower()
    if turn_manager_type == "round_robin":
        return RoundRobbin()
    elif turn_manager_type == "random_weighted":
        return RandomWeighted(config=config)
    else:
        raise ValueError(
            f"There is no turn manager option called {turn_manager_type}\n"
            + "Valid values: round_robin, random_weighted"
        )
