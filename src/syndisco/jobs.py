import collections
import datetime
import json
import logging
import uuid
import copy
import textwrap
import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from . import actors, turn_manager
from . import _file_util


logger = logging.getLogger(Path(__file__).name)


class DiscussionLogs:
    """
    Stores and serializes the log entries of a discussion.

    Each entry is a dict with keys ``name``, ``text``, and ``model``.
    The class can be constructed incrementally via :meth:`append`, or
    built all at once from plain lists via the :meth:`from_lists`
    class method.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, str]] = []

    @classmethod
    def from_lists(
        cls,
        names: list[str],
        texts: list[str],
        models: list[str] | None = None,
    ) -> "DiscussionLogs":
        """
        Build a :class:`DiscussionLogs` from parallel lists of names and
        texts, optionally with model labels.

        :param names: The username for each entry.
        :type names: list[str]
        :param texts: The message text for each entry.
        :type texts: list[str]
        :param models: The model label for each entry.  Defaults to
            ``"hardcoded"`` for every entry when *None*.
        :type models: list[str] | None, optional
        :raises ValueError: if *names* and *texts* (and, when provided,
            *models*) differ in length.
        :return: A populated :class:`DiscussionLogs` instance.
        :rtype: DiscussionLogs
        """
        if len(names) != len(texts):
            raise ValueError(
                f"names and texts must have the same length "
                f"(got {len(names)} and {len(texts)})."
            )
        if models is not None and len(models) != len(names):
            raise ValueError(
                f"models must have the same length as names and texts "
                f"(got {len(models)} vs {len(names)})."
            )

        instance = cls()
        resolved_models = (
            models if models is not None else ["hardcoded"] * len(names)
        )
        for name, text, model in zip(names, texts, resolved_models):
            instance.append(name=name, text=text, model=model)
        return instance

    def append(self, *, name: str, text: str, model: str) -> None:
        """
        Add a single log entry.

        :param name: The username of the speaker.
        :type name: str
        :param text: The message text.
        :type text: str
        :param model: The model (or ``"hardcoded"`` for seed entries).
        :type model: str
        """
        self._entries.append({"name": name, "text": text, "model": model})

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self._entries[index]

    def to_list(self) -> list[dict[str, str]]:
        """Return a shallow copy of the raw entry list."""
        return list(self._entries)

    def to_dict(
        self,
        *,
        extra: dict[str, Any] | None = None,
        timestamp_format: str = "%y-%m-%d-%H-%M",
    ) -> dict[str, Any]:
        """
        Serialize the logs to a dict, optionally merging caller-supplied
        metadata via *extra*.

        :param extra: Additional key/value pairs to include in the output
            dict.  Values in *extra* take precedence over the default keys
            (``timestamp``, ``logs``).
        :type extra: dict[str, Any] | None, optional
        :param timestamp_format: strftime format for the ``timestamp``
            field, defaults to ``"%y-%m-%d-%H-%M"``.
        :type timestamp_format: str, optional
        :return: A serializable dict representation of the logs.
        :rtype: dict[str, Any]
        """
        base: dict[str, Any] = {
            "timestamp": datetime.datetime.now().strftime(timestamp_format),
            "logs": self.to_list(),
        }
        if extra:
            base.update(extra)
        return base

    def to_json_file(
        self,
        output_path: str | Path,
        *,
        extra: dict[str, Any] | None = None,
        timestamp_format: str = "%y-%m-%d-%H-%M",
    ) -> None:
        """
        Write the logs (and any *extra* metadata) to a JSON file.

        :param output_path: Destination path for the JSON file.
        :type output_path: str | Path
        :param extra: Additional metadata to embed alongside the log
            entries (forwarded to :meth:`to_dict`).
        :type extra: dict[str, Any] | None, optional
        :param timestamp_format: strftime format for the timestamp field.
        :type timestamp_format: str, optional
        """
        _file_util.dict_to_json(
            self.to_dict(extra=extra, timestamp_format=timestamp_format),
            output_path,
        )

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


# No superclass because the shared method names between the
# Discussion and Annotation classes is coincidental.
class Discussion:
    """
    A job conducting a discussion between different actors
    (:class:`actors.Actor`).
    """

    def __init__(
        self,
        next_turn_manager: turn_manager.TurnManager,
        users: list[actors.Actor],
        history_context_len: int = 5,
        conv_len: int = 5,
        seed_opinions: list[str] | None = None,
        seed_opinion_usernames: list[str] | None = None,
    ) -> None:
        """
        Construct the framework for a conversation to take place.

        :param turn_manager: an object handling the speaker priority of the
            participants
        :type turn_manager: turn_manager.TurnManager
        :param users: A list of discussion participants
        :type users: list[actors.Actor]
        :param history_context_len: How many prior messages are included
            in the LLM's prompt as context, defaults to 5
        :type history_context_len: int, optional
        :param conv_len: The total length of the conversation
            (how many times each actor will be prompted), defaults to 5
        :type conv_len: int, optional
        :param seed_opinions: The first hardcoded comments to start the
            discussion with.  Inserted top-to-bottom in list order.
        :type seed_opinions: list[str], optional
        :param seed_opinion_usernames: The username for each seed opinion.
            Sampled randomly (without replacement) when *None*.
        :type seed_opinion_usernames: list[str], optional
        :raises ValueError: if the number of seed opinions and seed
            opinion users differ, or if seed opinions exceed
            *history_context_len*.
        """
        users = copy.copy(users)
        self.username_user_map = {user.get_name(): user for user in users}

        self.next_turn_manager = next_turn_manager

        # used only during export, tags underlying models
        self.user_types = [
            type(user).__name__ for user in self.username_user_map
        ]
        self.conv_len = conv_len

        # unique id for each conversation
        self.id = uuid.uuid4()

        # keep a limited context of the conversation to feed to the models
        self.ctx_len = history_context_len
        self.ctx_history: collections.deque[str] = collections.deque(
            maxlen=history_context_len
        )

        # all persistent log state is owned by DiscussionLogs
        self.logs = DiscussionLogs()

        self.seed_opinions = seed_opinions or []
        self.seed_opinion_usernames = seed_opinion_usernames

    def begin(self, verbose: bool = True) -> None:
        self.next_turn_manager.set_names(list(self.username_user_map.keys()))

        if len(self.logs) != 0:
            raise RuntimeError(
                "This conversation has already been concluded, "
                "create a new Discussion object."
            )

        self._add_seed_opinions(verbose)

        # begin main conversation
        for _ in tqdm(range(self.conv_len)):
            speaker_name = self.next_turn_manager.next()
            actor = self.username_user_map[speaker_name]
            res = actor.speak(list(self.ctx_history))

            if len(res.strip()) != 0:
                self._archive_response(actor, res, verbose)

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
        extra = {
            "id": str(self.id),
            "users": [
                user.get_name() for user in self.username_user_map.values()
            ],
            "user_prompts": [
                user.describe() for user in self.username_user_map.values()
            ],
            "ctx_length": self.ctx_len,
        }
        return self.logs.to_dict(
            extra=extra, timestamp_format=timestamp_format
        )

    def to_json_file(self, output_path: str | Path) -> None:
        """
        Export the data and metadata of the conversation as a JSON file.

        :param output_path: the path for the exported file
        :type output_path: str | Path
        """
        extra = {
            "id": str(self.id),
            "users": [
                user.get_name() for user in self.username_user_map.values()
            ],
            "user_prompts": [
                user.describe() for user in self.username_user_map.values()
            ],
            "ctx_length": self.ctx_len,
        }
        self.logs.to_json_file(output_path, extra=extra)

    def _add_seed_opinions(self, verbose: bool) -> None:
        if len(self.seed_opinions) > 0:
            usernames = self.seed_opinion_usernames
            if usernames is None:
                if len(self.seed_opinions) > len(self.username_user_map):
                    raise ValueError(
                        "Not enough users to assign unique usernames "
                        "for seed opinions."
                    )
                usernames = random.sample(
                    list(self.username_user_map.keys()),
                    len(self.seed_opinions),
                )

            for username, comment in zip(usernames, self.seed_opinions):
                seed_user = actors.Actor(
                    model=None,  # type: ignore
                    persona=actors.Persona(username=username),
                    context="",
                    instructions="",
                    actor_type=actors.ActorType.USER,
                )
                if comment.strip() != "":
                    self._archive_response(seed_user, comment, verbose=verbose)

    def _archive_response(
        self, user: actors.Actor, comment: str, verbose: bool
    ) -> None:
        """
        Persist *comment* to the log and the rolling context window.

        :param user: The user who created the new comment.
        :type user: actors.Actor
        :param comment: The new comment.
        :type comment: str
        :param verbose: Whether to print the comment to stdout.
        :type verbose: bool
        """
        model_name = (
            user.model.get_name() if user.model is not None else "hardcoded"
        )
        self.logs.append(name=user.get_name(), text=comment, model=model_name)
        self._add_comment_to_history(user, comment, verbose)

    def _add_comment_to_history(
        self, user: actors.Actor, comment: str, verbose: bool
    ) -> None:
        """
        Add *comment* to the rolling context window shown to future
        participants, and optionally print it.

        :param user: The user who created the new comment.
        :type user: actors.Actor
        :param comment: The new comment.
        :type comment: str
        :param verbose: Whether to print the comment to stdout.
        :type verbose: bool
        """
        formatted_res = _format_chat_message(user.get_name(), comment)
        self.ctx_history.append(formatted_res)
        if verbose:
            print(formatted_res, "\n")

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class Annotation:
    """
    An annotation job modelled as a discussion between the system writing
    the logs of a finished discussion, and the LLM Annotator.
    """

    def __init__(
        self,
        annotator: actors.Actor,
        conv_logs_path: str | Path,
        history_ctx_len: int = 2,
    ):
        """
        Create an annotation job.

        :param annotator: The annotator.
        :type annotator: actors.Actor
        :param conv_logs_path: Path to the JSON file containing the
            discussion logs.
        :type conv_logs_path: str | Path
        :param history_ctx_len: How many previous comments the annotator
            will remember, defaults to 2.
        :type history_ctx_len: int, optional
        """
        self.annotator = annotator
        self.history_ctx_len = history_ctx_len
        self.logs = DiscussionLogs()

        with open(conv_logs_path, "r", encoding="utf8") as fin:
            self.conv_data_dict = json.load(fin)

    def begin(self, verbose: bool = True) -> None:
        """
        Begin the conversation-modelled annotation job.

        :param verbose: Whether to print annotation results to the console,
            defaults to True.
        :type verbose: bool, optional
        """
        ctx_history: collections.deque[str] = collections.deque(
            maxlen=self.history_ctx_len
        )

        for message_data in tqdm(self.conv_data_dict["logs"]):
            username = message_data["name"]
            message = message_data["text"]

            formatted_message = _format_chat_message(username, message)
            ctx_history.append(formatted_message)
            annotation = self.annotator.speak(list(ctx_history))
            self.logs.append(
                name=username,
                text=annotation,
                model=self.annotator.model.get_name(),
            )

            if verbose:
                print(textwrap.fill(formatted_message))
                print(annotation)

    def to_dict(
        self, timestamp_format: str = "%y-%m-%d-%H-%M"
    ) -> dict[str, Any]:
        """
        Get a dictionary view of the annotation data and metadata.

        :param timestamp_format: strftime format for the timestamp field,
            defaults to ``"%y-%m-%d-%H-%M"``.
        :type timestamp_format: str, optional
        :return: a dict representing the annotation run.
        :rtype: dict[str, Any]
        """
        extra = {
            "conv_id": str(self.conv_data_dict["id"]),
            "annotator_model": self.annotator.model.get_name(),
            "annotator_prompt": self.annotator.describe(),
            "ctx_length": self.history_ctx_len,
        }
        return self.logs.to_dict(
            extra=extra, timestamp_format=timestamp_format
        )

    def to_json_file(self, output_path: str | Path) -> None:
        """
        Export the annotation data and metadata as a JSON file.

        :param output_path: the path for the exported file.
        :type output_path: str | Path
        """
        extra = {
            "conv_id": str(self.conv_data_dict["id"]),
            "annotator_model": self.annotator.model.get_name(),
            "annotator_prompt": self.annotator.describe(),
            "ctx_length": self.history_ctx_len,
        }
        self.logs.to_json_file(output_path, extra=extra)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


def _format_chat_message(username: str, message: str) -> str:
    """
    Create a prompt-friendly/console-friendly string representing a message
    made by a user.

    :param username: the name of the user who made the post.
    :type username: str
    :param message: the message that was posted.
    :type message: str
    :return: a formatted string containing both username and message.
    :rtype: str
    """
    if len(message.strip()) != 0:
        # Prefix with "User X posted:" so the model doesn't confuse it
        # with the instruction prompt.
        wrapped_res = textwrap.fill(message, 70)
        formatted_res = f"User {username} posted:\n{wrapped_res}"
    else:
        formatted_res = ""

    return formatted_res
