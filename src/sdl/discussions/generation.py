"""
Runtime class which saves synthetic discussion experiment 
configs at runtime and is responsible for executing it.
"""
import collections
import datetime
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from ..backend import actors, turn_manager
from ..util import output_util, file_util


logger = logging.getLogger(Path(__file__).name)


class Conversation:
    """
    A class conducting a conversation between different actors (:class:`actors.Actor`).
    Only one object should be used for a given conversation.
    """

    def __init__(
        self,
        next_turn_manager: turn_manager.TurnManager,
        users: list[actors.LLMUser],
        moderator: Optional[actors.LLMUser] = None,
        history_context_len: int = 5,
        conv_len: int = 5,
        seed_opinion: str = "",
        seed_opinion_user: str = "",
    ) -> None:
        """
        Construct the framework for a conversation to take place.

        :param turn_manager: an object handling the speaker priority of the users
        :type turn_manager: turn_manager.TurnManager
        :param users: A list of discussion participants
        :type users: list[actors.Actor]
        :param moderator: An actor tasked with moderation if not None, 
        can speak at any point in the conversation,
         defaults to None
        :type moderator: actors.Actor | None, optional
        :param history_context_len: How many prior messages are included 
        to the LLMs prompt as context, defaults to 5
        :type history_context_len: int, optional
        :param conv_len: The total length of the conversation 
        (how many times each actor will be prompted),
         defaults to 5
        :type conv_len: int, optional
        :param seed_opinion: The first hardcoded comments to 
        start the conversation with
        :type seed_opinion: str, optional
        :param seed_opinion_user: The username for the seed opinion
        :type seed_opinion_user: int, optional
        :raises ValueError: if the number of seed opinions and seed 
        opinion users are different, or
        if the number of seed opinions exceeds history_context_len
        """
        # just to satisfy the type checker
        self.next_turn_manager = next_turn_manager
        self.username_user_map = {user.get_name(): user for user in users}
        # used during export, in order to keep information about the underlying models
        self.user_types = [type(user).__name__ for user in self.username_user_map]
        self.moderator = moderator
        self.conv_len = conv_len
        # unique id for each conversation, generated for persistence purposes
        self.id = uuid.uuid4()

        self.conv_logs = []
        # keep a limited context of the conversation to feed to the models
        self.ctx_len = history_context_len
        self.ctx_history = collections.deque(maxlen=history_context_len)

        self.seed_opinion_user = seed_opinion_user
        self.seed_opinion = seed_opinion

    def begin_conversation(self, verbose: bool = True) -> None:
        """
        Begin the conversation between the actors.
        :param verbose: whether to print the messages on the screen 
        as they are generated, defaults to True
        :type verbose: bool, optional
        :raises RuntimeError: if the object has already been used to generate a conversation
        """
        if len(self.conv_logs) != 0:
            raise RuntimeError(
                "This conversation has already been concluded, create a new Conversation object."
            )

        if self.seed_opinion.strip() != "":
            # create first "seed" opinion
            seed_user = actors.LLMUser(
                model=None,  # type: ignore
                name=self.seed_opinion_user,
                attributes=[],
                context="",
                instructions="",
            )
            self._archive_response(seed_user, self.seed_opinion, verbose=verbose)
        else:
            logger.info("No seed opinion provided.")

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

    def to_dict(self, timestamp_format: str = "%y-%m-%d-%H-%M") -> dict[str, Any]:
        """
        Get a dictionary view of the data and metadata contained in the conversation.

        :param timestamp_format: the format for the conversation's creation time,
         defaults to "%y-%m-%d-%H-%M"
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
            "ctx_length": self.ctx_len,
            "logs": self.conv_logs,
        }

    def to_json_file(self, output_path: str | Path) -> None:
        """
        Export the data and metadata of the conversation as a json file.
        Convenience function equivalent to json.dump(self.to_dict(), output_path)

        :param output_path: the path for the exported file
        :type output_path: str
        """
        file_util.ensure_parent_directories_exist(output_path)

        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(self.to_dict(), fout, indent=4)

    def _archive_response(
        self, user: actors.LLMUser, comment: str, verbose: bool
    ) -> None:
        """Save the new comment to discussion output, 
        to discussion history for other users to see, maybe print it on screen.

        :param user: The user who created the new comment.
        :type user: actors.LLMUser
        :param comment: The new comment.
        :type comment: str
        :param verbose: Whether to print the comment to stdout
        :type verbose: bool
        """
        self._log_comment(user, comment)
        self._add_comment_to_history(user, comment, verbose)

    def _log_comment(self, user: actors.LlmActor, comment: str) -> None:
        """Save new comment to the output history.

        :param user: The user who created the new comment
        :type user: actors.LlmActor
        :param comment: The new comment
        :type comment: str
        """
        model_name = user.model.name if user.model is not None else "hardcoded"
        artifact = {"name": user.name, "text": comment, "model": model_name}
        self.conv_logs.append(artifact)

    def _add_comment_to_history(
        self, user: actors.LlmActor, comment: str, verbose: bool
    ) -> None:
        """Add new comment to the discussion history, 
        so it can be shown to the other participants in the future.

        :param user: The user who created the new comment
        :type user: actors.LlmActor
        :param comment: The new comment
        :type comment: str
        :param verbose: Whether to print the comment to stdout
        :type verbose: bool
        """
        formatted_res = output_util.format_chat_message(user.name, comment)
        self.ctx_history.append(formatted_res)

        if verbose:
            print(formatted_res, "\n")

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
