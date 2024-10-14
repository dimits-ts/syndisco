import collections
import dataclasses
import datetime
import json
import uuid
from typing import Any

import lib.actors
import lib.models
import lib.util


class Conversation:
    """
    A class conducting a conversation between different actors (:class:`lib.actors.Actor`).
    Only one object should be used for a given conversation.
    """

    def __init__(
            self,
            users: list[lib.actors.IActor],
            moderator: lib.actors.IActor | None = None,
            history_context_len: int = 5,
            conv_len: int = 5,
    ) -> None:
        """
        Construct the framework for a conversation to take place.

        :param users: A list of discussion participants
        :type users: list[lib.actors.Actor]
        :param moderator: An actor tasked with moderation if not None, can speak at any point in the conversation,
         defaults to None
        :type moderator: lib.actors.Actor | None, optional
        :param history_context_len: How many prior messages are included to the LLMs prompt as context, defaults to 5
        :type history_context_len: int, optional
        :param conv_len: The total length of the conversation (how many times each actor will be prompted),
         defaults to 5
        :type conv_len: int, optional
        """
        self.users = users
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
            for user in self.users:
                self._actor_turn(user, verbose)

                # TODO: refactor for more than 2 users
                # if one of the two users stop speaking
                if len(self.conv_logs[-1][1].strip()) == 0:
                    return

                if self.moderator is not None:
                    self._actor_turn(self.moderator, verbose)

    def _actor_turn(self, actor: lib.actors.IActor, verbose: bool) -> None:
        """
        Prompt the actor to speak and record his response accordingly.

        :param actor: the actor to speak, can be both a user and a moderator
        :type actor: lib.actors.Actor
        :param verbose: whether to also print the message on the screen
        :type verbose: bool
        """
        res = actor.speak(list(self.ctx_history))
        formatted_res = lib.util.format_chat_message(actor.get_name(), res)

        if verbose:
            print(formatted_res)

        self.ctx_history.append(formatted_res)
        self.conv_logs.append((actor.get_name(), res))

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
            "users": [user.get_name() for user in self.users],
            "user_types": [type(user).__name__ for user in self.users],
            "moderator": (
                self.moderator.get_name() if self.moderator is not None else None
            ),
            "moderator_type": (
                type(self.moderator).__name__ if self.moderator is not None else None
            ),
            "user_prompts": [user.describe() for user in self.users],
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
        lib.util.ensure_parent_directories_exist(output_path)

        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(self.to_dict(), fout, indent=4)

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


@dataclasses.dataclass
class LLMConvData:
    """
    A dataclass responsible for serializing and deserializing data needed to construct a :class:`Conversation`.
    """
    context: str
    user_names: list[str]
    user_attributes: list[list[str]]
    user_instructions: str
    conv_len: int = 4
    history_ctx_len: int = 4
    moderator_name: str | None = None
    moderator_attributes: list[str] | None = None
    moderator_instructions: str | None = None

    def __post_init__(self):
        assert len(self.user_names) == len(
            self.user_attributes
        ), "Number of actor names and actor attribute lists must be the same"

    @staticmethod
    def from_json_file(input_file_path: str):
        """
        Construct a LLMConvData instance according to a serialized .json file.

        :param input_file_path: The path to the serialized .json file
        :type input_file_path: str
        :return: A LLMConvData instance containing the information from the file
        :rtype: LLMConvData
        """
        with open(input_file_path, "r", encoding="utf8") as fin:
            data_dict = json.load(fin)

        # code from https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary
        field_set = {f.name for f in dataclasses.fields(LLMConvData) if f.init}
        filtered_arg_dict = {k: v for k, v in data_dict.items() if k in field_set}
        return LLMConvData(**filtered_arg_dict)

    def to_json_file(self, output_path: str) -> None:
        """
        Serialize the data to a .json file.

        :param output_path: The path of the new file
        :type output_path: str
        """
        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(dataclasses.asdict(self), fout, indent=4)


class LLMConvGenerator:
    """
    A class responsible for creating a :class:`Conversation` from the conversation data (:class:`LLMConvData`)
    and a model (:class:`lib.models.LlamaModel`).
    """

    def __init__(self,
                 data: LLMConvData,
                 user_model: lib.models.LlamaModel,
                 moderator_model: lib.models.LlamaModel | None,
                 ):
        """
        Initialize the generator.

        :param data: The deserialized conversation input data
        :type data: LLMConvData
        :param user_model: The model used for the users to talk
        :type user_model: tasks.models.LlamaModel
        :param moderator_model: The model used for the moderator to talk, if he exists
        :type moderator_model: tasks.models.LlamaModel | None
        """
        assert user_model is not None, "User model cannot be None"
        assert not (moderator_model is None and data.moderator_name is not None), ("Moderator agent was not given a "
                                                                                   "model.")
        self.user_model = user_model
        self.moderator_model = moderator_model
        self.data = data

    def produce_conversation(self) -> Conversation:
        """
        Generate a conversation.

        :return: An initialized Conversation instance.
        :rtype: Conversation
        """
        user_list = []

        for i in range(len(self.data.user_names)):
            user_list.append(lib.actors.LLMUser(model=self.user_model,
                                                name=self.data.user_names[i],
                                                role="chat user",
                                                attributes=self.data.user_attributes[i],
                                                context=self.data.context,
                                                instructions=self.data.user_instructions))
        if self.data.moderator_name is not None:
            moderator = lib.actors.LLMUser(model=self.moderator_model,
                                           name=self.data.moderator_name,
                                           role="chat moderator",
                                           attributes=self.data.moderator_attributes,
                                           context=self.data.context,
                                           instructions=self.data.moderator_instructions)
        else:
            moderator = None

        conversation = Conversation(users=user_list,
                                    moderator=moderator,
                                    history_context_len=self.data.history_ctx_len,
                                    conv_len=self.data.conv_len)
        return conversation
