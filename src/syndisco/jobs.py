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

import collections
import datetime
import json
import logging
import uuid
import textwrap
from pathlib import Path
from typing import Any, Optional

from .backend import actors, turn_manager
from .util import file_util


logger = logging.getLogger(Path(__file__).name)


# No superclass because the shared method names between the classes
# is coincidental


class Discussion:
    """
    A job conducting a discussion between different actors
    (:class:`actors.Actor`).
    """

    def __init__(
        self,
        next_turn_manager: turn_manager.TurnManager,
        users: list[actors.LLMActor],
        moderator: Optional[actors.LLMActor] = None,
        history_context_len: int = 5,
        conv_len: int = 5,
        seed_opinion: str = "",
        seed_opinion_user: str = "",
    ) -> None:
        """
        Construct the framework for a conversation to take place.

        :param turn_manager: an object handling the speaker priority of the
        participants
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
        self.username_user_map = {user.get_name(): user for user in users}

        next_turn_manager.set_names([user.get_name() for user in users])
        self.next_turn_manager = next_turn_manager

        # used only during export, tags underlying models
        self.user_types = [
            type(user).__name__ for user in self.username_user_map
        ]

        self.moderator = moderator
        self.conv_len = conv_len

        # unique id for each conversation
        self.id = uuid.uuid4()

        # keep a limited context of the conversation to feed to the models
        self.ctx_len = history_context_len
        self.ctx_history = collections.deque(maxlen=history_context_len)
        self.conv_logs = []

        self.seed_opinion_user = seed_opinion_user
        self.seed_opinion = seed_opinion

    def begin(self, verbose: bool = True) -> None:
        """
        Begin the discussion between the actors.

        :param verbose: whether to print the messages on the screen
            as they are generated, defaults to True
        :type verbose: bool, optional
        :raises RuntimeError: if the object has already been used to generate
            a conversation
        """
        if len(self.conv_logs) != 0:
            raise RuntimeError(
                "This conversation has already been concluded, "
                "create a new Discussion object."
            )

        if self.seed_opinion.strip() != "":
            # create first "seed" opinion
            seed_user = actors.LLMActor(
                model=None,  # type: ignore
                name=self.seed_opinion_user,
                attributes=[],
                context="",
                instructions="",
                actor_type=actors.ActorType.USER,
            )
            self._archive_response(
                seed_user, self.seed_opinion, verbose=verbose
            )
        else:
            logger.info("No seed opinion provided.")

        # begin generation
        for _ in range(self.conv_len):
            speaker_name = self.next_turn_manager.next()
            actor = self.username_user_map[speaker_name]
            res = actor.speak(list(self.ctx_history))

            # if nothing was said, do not include it in history
            if len(res.strip()) != 0:
                self._archive_response(actor, res, verbose)

                # if something was said and there is a moderator, prompt him
                if self.moderator is not None:
                    res = self.moderator.speak(list(self.ctx_history))
                    self._archive_response(self.moderator, res, verbose)

    def to_dict(
        self, timestamp_format: str = "%y-%m-%d-%H-%M"
    ) -> dict[str, Any]:
        """
        Get a dictionary view of the data and metadata contained in the
        discussion.

        :param timestamp_format: the format for the conversation's creation
            time, defaults to "%y-%m-%d-%H-%M"
        :type timestamp_format: str, optional
        :return: a dict representing the conversation
        :rtype: dict[str, Any]
        """
        return {
            "id": str(self.id),
            "timestamp": datetime.datetime.now().strftime(timestamp_format),
            "users": [
                user.get_name() for user in self.username_user_map.values()
            ],
            "moderator": (
                self.moderator.get_name()
                if self.moderator is not None
                else None
            ),
            "user_prompts": [
                user.describe() for user in self.username_user_map.values()
            ],
            "moderator_prompt": (
                self.moderator.describe()
                if self.moderator is not None
                else None
            ),
            "ctx_length": self.ctx_len,
            "logs": self.conv_logs,
        }

    def to_json_file(self, output_path: str | Path) -> None:
        """
        Export the data and metadata of the conversation as a json file.

        :param output_path: the path for the exported file
        :type output_path: str
        """
        file_util.dict_to_json(self.to_dict(), output_path)

    def _archive_response(
        self, user: actors.LLMActor, comment: str, verbose: bool
    ) -> None:
        """
        Save the new comment to discussion output,
        to discussion history for other users to see, maybe print it on screen.

        :param user: The user who created the new comment.
        :type user: actors.LLMActor
        :param comment: The new comment.
        :type comment: str
        :param verbose: Whether to print the comment to stdout
        :type verbose: bool
        """
        self._log_comment(user, comment)
        self._add_comment_to_history(user, comment, verbose)

    def _log_comment(self, user: actors.LLMActor, comment: str) -> None:
        """
        Save new comment to the output history.

        :param user: The user who created the new comment
        :type user: actors.LLMActor
        :param comment: The new comment
        :type comment: str
        """
        model_name = user.model.name if user.model is not None else "hardcoded"
        artifact = {"name": user.name, "text": comment, "model": model_name}
        self.conv_logs.append(artifact)

    def _add_comment_to_history(
        self, user: actors.LLMActor, comment: str, verbose: bool
    ) -> None:
        """
        Add new comment to the discussion history,
            so it can be shown to the other participants in the future.

        :param user: The user who created the new comment
        :type user: actors.LLMActor
        :param comment: The new comment
        :type comment: str
        :param verbose: Whether to print the comment to stdout
        :type verbose: bool
        """
        formatted_res = _format_chat_message(user.name, comment)
        self.ctx_history.append(formatted_res)

        if verbose:
            print(formatted_res, "\n")

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class Annotation:
    """
    An annotation job modelled as a discussion between the system writing the
    logs of a finished discussion, and the LLM Annotator.
    """

    def __init__(
        self,
        annotator: actors.LLMActor,
        conv_logs_path: str | Path,
        include_moderator_comments: bool,
        history_ctx_len: int = 2,
    ):
        """
        Create an annotation job.
        The annotation is modelled as a conversation between the system and
        the annotator.

        :param annotator: The annotator
        :type annotator: actors.IActor
        :param conv_logs_path: The path to the file containing the
        discussion logs in JSON format
        :type conv_logs_path: str | Path
        :param include_moderator_comments: Whether to annotate moderator
        comments, and include them
        in conversational context when annotating user responses.
        :type include_moderator_comments: bool
        :param history_ctx_len: How many previous comments the annotator will
        remember, defaults to 4
        :type history_ctx_len: int, optional
        """
        self.annotator = annotator
        self.history_ctx_len = history_ctx_len
        self.include_moderator_comments = include_moderator_comments
        self.annotation_logs = []

        with open(conv_logs_path, "r", encoding="utf8") as fin:
            self.conv_data_dict = json.load(fin)

    def begin(self, verbose=True) -> None:
        """
        Begin the conversation-modelled annotation job.

        :param verbose: whether to print the results of the annotation to the
            console, defaults to True
        :type verbose: bool, optional
        """
        ctx_history = collections.deque(maxlen=self.history_ctx_len)

        for message_data in self.conv_data_dict["logs"]:
            username = message_data["name"]
            message = message_data["text"]

            # do not include moderator comments in annotation ctx if told so
            if "moderator" in username:
                if not self.include_moderator_comments:
                    continue

            formatted_message = _format_chat_message(username, message)
            ctx_history.append(formatted_message)
            annotation = self.annotator.speak(list(ctx_history))
            self.annotation_logs.append((message, annotation))

            if verbose:
                print(textwrap.fill(formatted_message))
                print(annotation)

    def to_dict(
        self, timestamp_format: str = "%y-%m-%d-%H-%M"
    ) -> dict[str, Any]:
        """
        Get a dictionary view of the data and metadata contained in
        the annotation.

        :param timestamp_format: the format for the conversation's creation
            time, defaults to "%y-%m-%d-%H-%M"
        :type timestamp_format: str, optional
        :return: a dict representing the conversation
        :rtype: dict[str, Any]
        """
        return {
            "conv_id": str(self.conv_data_dict["id"]),
            "timestamp": datetime.datetime.now().strftime(timestamp_format),
            "annotator_model": self.annotator.model.name,
            "annotator_prompt": self.annotator.describe(),
            "ctx_length": self.history_ctx_len,
            "logs": self.annotation_logs,
        }

    def to_json_file(self, output_path: str | Path) -> None:
        """
        Export the data and metadata of the conversation as a json file.

        :param output_path: the path for the exported file
        :type output_path: str
        """
        file_util.dict_to_json(self.to_dict(), output_path)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


def _format_chat_message(username: str, message: str) -> str:
    """
    Create a prompt-friendly/console-friendly string representing a message
    made by a user.

    :param username: the name of the user who made the post
    :type username: str
    :param message: the message that was posted
    :type message: str
    :return: a formatted string containing both username and his message
    :rtype: str
    """
    if len(message.strip()) != 0:
        # append name of actor to his response
        # "user x posted" important for the model to not confuse it
        # with the instruction prompt
        wrapped_res = textwrap.fill(message, 70)
        formatted_res = f"User {username} posted:\n{wrapped_res}"
    else:
        formatted_res = ""

    return formatted_res
