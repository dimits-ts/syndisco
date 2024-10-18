import collections

import datetime
import json
import uuid
from typing import Any

from sdl import actors
from sdl import util
from sdl import turn_manager


class Conversation:
    """
    A class conducting a conversation between different actors (:class:`actors.Actor`).
    Only one object should be used for a given conversation.
    """

    def __init__(
            self,
            turn_manager: turn_manager.TurnManager,
            users: list[actors.IActor],
            moderator: actors.IActor | None = None,
            history_context_len: int = 5,
            conv_len: int = 5,
    ) -> None:
        """
        Construct the framework for a conversation to take place.

        :param turn_manager: an object handling the speaker priority of the users
        :type turn_manager: turn_manager.TurnManager
        :param users: A list of discussion participants
        :type users: list[actors.Actor]
        :param moderator: An actor tasked with moderation if not None, can speak at any point in the conversation,
         defaults to None
        :type moderator: actors.Actor | None, optional
        :param history_context_len: How many prior messages are included to the LLMs prompt as context, defaults to 5
        :type history_context_len: int, optional
        :param conv_len: The total length of the conversation (how many times each actor will be prompted),
         defaults to 5
        :type conv_len: int, optional
        """
        # just to satisfy the type checker
        self.next_turn_manager = turn_manager
        self.users = {user.get_name(): user for user in users}
        # used during export, in order to keep information about the underlying models
        self.user_types = [type(user).__name__ for user in self.users]
        self.moderator = moderator
        self.conv_len = conv_len
        # unique id for each conversation, generated for persistence purposes
        self.id = uuid.uuid4()

        self.conv_logs = []
        # keep a limited context of the conversation to feed to the models
        self.ctx_history = collections.deque(maxlen=history_context_len)

    def begin_conversation(self, verbose: bool = True) -> None:
        """
        Begin the conversation between the actors.

        :param verbose: whether to print the messages on the screen as they are generated, defaults to True
        :type verbose: bool, optional
        :raises RuntimeError: if the object has already been used to generate a conversation
        """
        if len(self.conv_logs) != 0:
            raise RuntimeError(
                "This conversation has already been concluded, create a new Conversation object."
            )

        for _ in range(self.conv_len):
            speaker_name = self.next_turn_manager.next_turn_username()
            res = self._actor_turn(self.users[speaker_name])
            
            # if nothing was said, do not include it in history
            if len(res.strip()) != 0:
                self._archive_response(speaker_name, res, verbose)

                #if something was said and there is a moderator, prompt him
                if self.moderator is not None:
                    res = self._actor_turn(self.moderator)
                    self._archive_response(self.moderator.get_name(), res, verbose)

    def _actor_turn(self, actor: actors.IActor) -> str:
        """
        Prompt the actor to speak and record his response accordingly.

        :param actor: the actor to speak, can be both a user and a moderator
        :type actor: actors.Actor
        :param verbose: whether to also print the message on the screen
        :type verbose: bool
        """
        res = actor.speak(list(self.ctx_history))
        formatted_res = util.format_chat_message(actor.get_name(), res)
        return formatted_res

    def _archive_response(self, username: str, response: str, verbose: bool) -> None:
        self.ctx_history.append(response)
        self.conv_logs.append((username, response))

        if verbose:
            print(response)

    def to_dict(self, timestamp_format: str = "%y-%m-%d-%H-%M") -> dict[str, Any]:
        """
        Get a dictionary view of the data and metadata contained in the conversation.

        :param timestamp_format: the format for the conversation's creation time, defaults to "%y-%m-%d-%H-%M"
        :type timestamp_format: str, optional
        :return: a dict representing the conversation
        :rtype: dict[str, Any]
        """
        return {
            "id": str(self.id),
            "timestamp": datetime.datetime.now().strftime(timestamp_format),
            "users": [user.get_name() for user in self.users.values()],
            "user_types": self.user_types,
            "moderator": (
                self.moderator.get_name() if self.moderator is not None else None
            ),
            "moderator_type": (
                type(self.moderator).__name__ if self.moderator is not None else None
            ),
            "user_prompts": [user.describe() for user in self.users.values()],
            "moderator_prompt": (
                self.moderator.describe() if self.moderator is not None else None
            ),
            "ctx_length": len(self.ctx_history),
            "logs": self.conv_logs,
        }

    def to_json_file(self, output_path: str):
        """
        Export the data and metadata of the conversation as a json file.
        Convenience function equivalent to json.dump(self.to_dict(), output_path)

        :param output_path: the path for the exported file
        :type output_path: str
        """
        util.ensure_parent_directories_exist(output_path)

        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(self.to_dict(), fout, indent=4)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


