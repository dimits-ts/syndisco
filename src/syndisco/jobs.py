import collections
import collections.abc
import datetime
import json
import logging as pylog
import copy
import textwrap
import random
import typing
from pathlib import Path

from tqdm.auto import tqdm

from . import actors, turn_manager
from . import _file_util


logger = pylog.getLogger(Path(__file__).name)


class Logs:
    """
    A mutable container for comments made in a discussion.

    Each entry is a dict with keys ``name``, ``text``, and ``model``.
    The class can be constructed incrementally via :meth:`append`, or
    built all at once from plain lists via the :meth:`from_lists`
    class method.
    """

    def __init__(self) -> None:
        self._entries: list[dict[str, str]] = []

    @classmethod
    def from_file(cls, path: str | Path) -> "Logs":
        """
        Load a :class:`DiscussionLogs` from a JSON file previously written
        by :meth:`to_json_file`.

        :param path: Path to the JSON file.
        :type path: str | Path
        :raises FileNotFoundError: if *path* does not exist.
        :raises ValueError: if the JSON does not match the expected schema.
        :return: A populated :class:`DiscussionLogs` instance.
        :rtype: DiscussionLogs
        """
        with open(path, "r", encoding="utf8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"File is not valid JSON: {e}") from e

        if "logs" not in data:
            raise ValueError("Missing required key 'logs' in JSON schema.")

        if not isinstance(data["logs"], list):
            raise ValueError("'logs' must be a list.")

        required_entry_keys = {"name", "text", "model"}
        for i, entry in enumerate(data["logs"]):
            missing = required_entry_keys - entry.keys()
            if missing:
                raise ValueError(
                    f"Log entry {i} is missing required keys: {missing}."
                )

        instance = Logs()
        for entry in data["logs"]:
            instance.append(
                name=entry["name"], text=entry["text"], model=entry["model"]
            )
        return instance

    def append(
        self, name: str, text: str, model: str = "hardcoded", prompt: str = ""
    ) -> None:
        """
        Add a single log entry.

        :param name: The username of the speaker.
        :type name: str
        :param text: The message text.
        :type text: str
        :param model: The model (or ``"hardcoded"`` for seed entries).
        :type model: str
        :param prompt:
            The prompt given to the LLM user that generated the response.
            Empty string if the user is not an LLM.
        :type prompt: str:
        """
        self._entries.append(
            {"name": name, "text": text, "model": model, "prompt": prompt}
        )

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self._entries[index]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Logs):
            return NotImplemented
        return self._entries == other._entries

    def to_list(self) -> list[dict[str, str]]:
        """Return a shallow copy of the raw entry list."""
        return list(self._entries)

    def to_dict(
        self,
        timestamp_format: str = "%y-%m-%d-%H-%M",
    ) -> dict[str, typing.Any]:
        """
        Serialize the logs to a dict.

        :param timestamp_format: strftime format for the ``timestamp``
            field, defaults to ``"%y-%m-%d-%H-%M"``.
        :type timestamp_format: str, optional
        :return: A serializable dict representation of the logs.
        :rtype: dict[str, Any]
        """
        export_dict: dict[str, typing.Any] = {
            "timestamp": datetime.datetime.now().strftime(timestamp_format),
            "logs": self.to_list(),
        }

        return export_dict

    def export(
        self,
        output_path: str | Path,
        timestamp_format: str = "%y-%m-%d-%H-%M",
    ) -> None:
        """
        Write the logs (and any *extra* metadata) to a JSON file.

        :param output_path: Destination path for the JSON file.
        :type output_path: str | Path
        :param timestamp_format: strftime format for the timestamp field.
        :type timestamp_format: str, optional
        """
        _file_util.dict_to_json(
            self.to_dict(timestamp_format=timestamp_format),
            output_path,
        )

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


class Discussion(collections.abc.Iterator[dict[str, str]]):
    """
    A job conducting a discussion between different actors
    (:class:`actors.Actor`).

    ``Discussion`` implements the iterator protocol: each call to
    :func:`next` prompts the next speaker and returns the resulting log
    entry.  This means it can be driven step-by-step::

        discussion = Discussion(...)
        for entry in discussion:          # full run
            print(entry["name"], entry["text"])

    or consumed one turn at a time::

        it = iter(discussion)
        first = next(it)
        second = next(it)

    :meth:`begin` is a convenience wrapper that exhausts the iterator
    while printing output, matching the original one-shot API.

    Because ``Discussion`` is its own iterator (``__iter__`` returns
    ``self``), it is single-pass: once ``StopIteration`` is raised the
    instance is exhausted and should not be reused.
    """

    def __init__(
        self,
        next_turn_manager: turn_manager.TurnManager,
        users: collections.abc.Iterable[actors.Actor],
        history_context_len: int = 5,
        conv_len: int = 5,
        seed_opinions: typing.Sequence[str] | None = None,
        seed_opinion_usernames: typing.Sequence[str] | None = None,
    ) -> None:
        """
        Construct the framework for a conversation to take place.

        :param next_turn_manager: An object handling the speaker order of
            the participants.
        :type next_turn_manager: turn_manager.TurnManager
        :param users: Any iterable containing the discussion participants.
        :type users: Iterable[actors.Actor]
        :param history_context_len: How many prior messages are included
            in the LLM's prompt as context, defaults to 5.
        :type history_context_len: int, optional
        :param conv_len: The total number of prompted turns (seed opinions
            do not count toward this), defaults to 5.
        :type conv_len: int, optional
        :param seed_opinions: Hardcoded opening comments inserted before
            the first prompted turn, top-to-bottom in list order.
        :type seed_opinions: Sequence[str], optional
        :param seed_opinion_usernames: The username for each seed opinion.
            Sampled randomly (without replacement) when *None*.
        :type seed_opinion_usernames: Sequence[str], optional
        :raises ValueError: if the number of seed opinions and seed
            opinion usernames differ, or if there are more seed opinions
            than participants.
        """
        users = copy.copy(users)
        if any([actor.is_annotator for actor in users]):
            raise ValueError(
                "Annotator users can not participate in discussions."
            )
        self._users = list(users)

        self._next_turn_manager = next_turn_manager
        self._next_turn_manager.set_actors(users)
        # iterator state
        self._steps_taken: int = 0

        self.conv_len = conv_len

        # keep a limited context of the conversation to feed to the models
        self._ctx_history: collections.deque[str] = collections.deque(
            maxlen=history_context_len
        )

        # all persistent log state is owned by DiscussionLogs
        self._logs = Logs()

        if (seed_opinions is None) ^ (seed_opinion_usernames is None):
            raise ValueError(
                "Seed opinions and their respective usernames should either "
                "be both None, or both defined."
            )

        if (
            seed_opinions is not None
            and seed_opinion_usernames is not None
            and len(seed_opinions) != len(seed_opinion_usernames)
        ):
            raise ValueError(
                f"Length of seed opinions ({len(seed_opinions)}) differs from "
                f"length of seed usernames ({len(seed_opinion_usernames)})"
            )

        if (
            seed_opinions is not None
            and seed_opinion_usernames is not None
            and any(
                [
                    entry is None
                    for entry in seed_opinions + seed_opinion_usernames
                ]
            )
        ):
            raise ValueError("Seed opinions and usernames should be non-None.")

        self._seed_opinions = seed_opinions or []
        self._seed_opinion_usernames = seed_opinion_usernames or []

        for opinion, name in zip(
            self._seed_opinions, self._seed_opinion_usernames
        ):
            self._archive_response(
                user=actors.Actor(name=name), comment=opinion
            )

    def __next__(self) -> dict[str, str]:
        """
        Prompt the next speaker and return the new log entry.

        Skips turns where the actor returns only whitespace (the step
        still counts toward ``conv_len``).  Raises :exc:`StopIteration`
        once ``conv_len`` steps have been taken.

        :return: The newly appended log entry (keys: ``name``, ``text``,
            ``model``).
        :rtype: dict[str, str]
        :raises StopIteration: when all ``conv_len`` turns are exhausted.
        """
        if self._steps_taken >= self.conv_len:
            raise StopIteration

        actor = self._next_turn_manager.next()
        res = actor.speak(list(self._ctx_history))
        self._steps_taken += 1

        if res.strip():
            self._archive_response(actor, res)
            return self._logs[-1]

        # Whitespace response: return a placeholder entry so the caller
        # always receives one value per next() call.
        return {"name": actor.get_actor_name(), "text": "", "model": ""}

    # Convenience one-shot API
    def begin(self, verbose: bool = True) -> None:
        """
        Run the entire discussion to completion, printing each entry when
        *verbose* is ``True``.

        This is a thin wrapper that exhausts the iterator via
        :func:`tqdm`, applying *verbose* printing along the way.

        :param verbose: Whether to print each comment to stdout,
            defaults to ``True``.
        :type verbose: bool, optional
        """
        for entry in tqdm(self, total=self.conv_len):
            if verbose and entry["text"]:
                formatted = _format_chat_message(entry["name"], entry["text"])
                print(formatted, "\n")

    def get_logs(self) -> Logs:
        """
        Get the logs of the discussion. Can be used to export the logs
        to a file.

        :return: A copy of the discussion logs.
        :rtype: DiscussionLogs
        """
        return copy.deepcopy(self._logs)

    def _add_seed_opinions(self) -> None:
        if len(self._seed_opinions) > 0:
            usernames = self._seed_opinion_usernames
            if usernames is None:
                if len(self._seed_opinions) > len(self._users):
                    raise ValueError(
                        "Not enough users to assign unique usernames "
                        "for seed opinions."
                    )
                usernames = random.sample(
                    list([user.get_actor_name() for user in self._users]),
                    len(self._seed_opinions),
                )

            for username, comment in zip(usernames, self._seed_opinions):
                seed_user = actors.Actor(
                    model=None,  # type: ignore
                    persona={"username": username},
                    context="",
                    instructions="",
                )
                if comment.strip() != "":
                    self._archive_response(seed_user, comment)

    def _archive_response(self, user: actors.Actor, comment: str) -> None:
        """
        Persist *comment* to the log and the rolling context window.

        :param user: The user who created the new comment.
        :type user: actors.Actor
        :param comment: The new comment.
        :type comment: str
        """
        model_name = (
            user._model.get_name() if user._model is not None else "hardcoded"
        )
        self._logs.append(
            name=user.get_actor_name(),
            text=comment,
            model=model_name,
            prompt=user.get_system_prompt(),
        )
        formatted = _format_chat_message(user.get_actor_name(), comment)
        self._ctx_history.append(formatted)


class Annotation:
    """
    An annotation job applied on a single discussion.

    Modelled as a discussion between the system writing
    the logs of a finished discussion, and a single LLM Annotator.
    """

    def __init__(
        self,
        annotator: actors.Actor,
        discussion_logs: Logs,
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
        if not annotator.is_annotator:
            annotator = copy.deepcopy(annotator)
            annotator.is_annotator = True

        self._annotator = annotator
        self._history_ctx_len = history_ctx_len
        self._discussion_logs = copy.deepcopy(discussion_logs)
        self._annotation_logs = Logs()

    def begin(self, verbose: bool = True) -> None:
        """
        Run annotation on the entire discussion, printing each entry when
        *verbose* is ``True``.

        :param verbose: Whether to print each comment to stdout,
            defaults to ``True``.
        :type verbose: bool, optional
        """
        ctx_history: collections.deque[str] = collections.deque(
            maxlen=self._history_ctx_len
        )

        for message_data in tqdm(self._discussion_logs):
            username = message_data["name"]
            message = message_data["text"]

            formatted_message = _format_chat_message(username, message)
            ctx_history.append(formatted_message)
            annotation = self._annotator.speak(list(ctx_history))
            self._annotation_logs.append(
                name=username,
                text=annotation,
                model=self._annotator.get_model_name(),
                prompt=self._annotator.get_user_prompt(),
            )

            if verbose:
                print(textwrap.fill(formatted_message))
                print(annotation)

    def get_logs(self) -> Logs:
        """
        Get the annotation logs for this job.

        :return:
            A copied DiscussionLogs object containing the annotator's
            judgements for each comment in the provided discussion.
        :rtype: DiscussionLogs
        """
        return copy.deepcopy(self._annotation_logs)


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
