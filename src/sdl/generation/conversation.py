import collections
import datetime
import json
import uuid
from typing import Any, Optional

from ..backend import actors, turn_manager
from ..util import output_util, file_util


class Conversation:
    """
    A class conducting a conversation between different actors (:class:`actors.Actor`).
    Only one object should be used for a given conversation.
    """

    def __init__(
        self,
        turn_manager: turn_manager.TurnManager,
        users: list[actors.LLMUser],
        moderator: Optional[actors.LLMUser] = None,
        history_context_len: int = 5,
        conv_len: int = 5,
        seed_opinions: list[str] = [],
        seed_opinion_users: list[str] = [],
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
        :param seed_opinions: The first hardcoded comments to start the conversation with
        :type seed_opinions: list[str], optional
        :param seed_opinion_users: The usernames of each seed opinion
        :type seed_opinion_users: int, optional
        :raises ValueError: if the number of seed opinions and seed opinion users are different, or
        if the number of seed opinions exceeds history_context_len
        """
        # just to satisfy the type checker
        self.next_turn_manager = turn_manager
        self.username_user_map = {user.get_name(): user for user in users}
        # used during export, in order to keep information about the underlying models
        self.user_types = [type(user).__name__ for user in self.username_user_map]
        self.moderator = moderator
        self.conv_len = conv_len
        # unique id for each conversation, generated for persistence purposes
        self.id = uuid.uuid4()

        self.conv_logs = []
        # keep a limited context of the conversation to feed to the models
        self.ctx_history = collections.deque(maxlen=history_context_len)

        if len(seed_opinion_users) != len(seed_opinions):
            raise ValueError(
                "Seed opinions and seed opinion users should have the same length."
            )

        if len(seed_opinions) > history_context_len:
            raise ValueError(
                "More seed opinions provided than model context length."
                "The first seed opinions will never be read by the model."
            )

        self.seed_opinion_users = seed_opinion_users
        self.seed_opinions = seed_opinions

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

        # hardcoded comments at the start of the conversation
        for seed_user_name, seed_opinion in zip(
            self.seed_opinion_users, self.seed_opinions
        ):
            # dummy LLMUser
            seed_user = actors.LLMUser(
                model=None,  # type: ignore
                name=seed_user_name,
                attributes=[],
                context="",
                instructions="",
            )
            self._archive_response(seed_user, seed_opinion, verbose=verbose)

        # begin generation
        for _ in range(self.conv_len):
            speaker_name = self.next_turn_manager.next_turn_username()
            actor = self.username_user_map[speaker_name]
            res = actor.speak(list(self.ctx_history))

            # if nothing was said, do not include it in history
            if len(res.strip()) != 0:
                self._archive_response(actor, res, verbose)

                # if something was said and there is a moderator, prompt him
                if self.moderator is not None:
                    res = self.moderator.speak(list(self.ctx_history))
                    self._archive_response(self.moderator, res, verbose)

    def _archive_response(
        self, user: actors.LlmActor, response: str, verbose: bool
    ) -> None:
        self._log_comment(user, response)
        self._add_comment_to_history(user, response, verbose)

    def _log_comment(self, user: actors.LlmActor, comment: str) -> None:
        artifact = {"name": user.name, "text": comment, "model": user.model}
        self.conv_logs.append(artifact)

    def _add_comment_to_history(
        self, user: actors.LlmActor, response: str, verbose: bool
    ) -> None:
        formatted_res = output_util.format_chat_message(user.name, response)
        self.ctx_history.append(formatted_res)

        if verbose:
            print(formatted_res)

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
            "users": [user.get_name() for user in self.username_user_map.values()],
            "moderator": (
                self.moderator.get_name() if self.moderator is not None else None
            ),
            "user_prompts": [
                user.describe() for user in self.username_user_map.values()
            ],
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
        file_util.ensure_parent_directories_exist(output_path)

        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(self.to_dict(), fout, indent=4)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
