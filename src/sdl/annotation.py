import dataclasses
import json
import collections
import datetime
import textwrap
from typing import Any

import lib.actors
import lib.models
import lib.util


# "...but if you look at conversations.py, this whole file violates DRY"
# Not really, for the most part the API is the same by convention, not because it
# uses the same functionality. You can replace the implementation here entirely
# without impacting conversations.py at all

class AnnotationConv:
    """
    An annotation job modelled as a conversation between the messages of a finished dialogue, and the LLM Annotator.
    """

    def __init__(self, annotator: lib.actors.IActor, conv_logs_path: str, history_ctx_len: int = 4):
        self.annotator = annotator
        self.history_ctx_len = history_ctx_len
        self.annotation_logs = []

        with open(conv_logs_path, "r", encoding="utf8") as fin:
            self.conv_data_dict = json.load(fin)

    def begin_annotation(self, verbose=True) -> None:
        """
        Begin the conversation-modelled annotation job.

        :param verbose: whether to print the results of the annotation to the console, defaults to True
        :type verbose: bool, optional
        """
        ctx_history = collections.deque(maxlen=self.history_ctx_len)

        for username, message in self.conv_data_dict["logs"]:
            formatted_message = lib.util.format_chat_message(username, message)
            ctx_history.append(formatted_message)
            annotation = self.annotator.speak(list(ctx_history))
            self.annotation_logs.append((message, annotation))

            if verbose:
                print(textwrap.fill(formatted_message))
                print(annotation)

    def to_dict(self, timestamp_format: str = "%y-%m-%d-%H-%M") -> dict[str, Any]:
        """
        Get a dictionary view of the data and metadata contained in the conversation.

        :param timestamp_format: the format for the conversation's creation time, defaults to "%y-%m-%d-%H-%M"
        :type timestamp_format: str, optional
        :return: a dict representing the conversation
        :rtype: dict[str, Any]
        """
        return {
            "conv_id": str(self.conv_data_dict["id"]),
            "timestamp": datetime.datetime.now().strftime(timestamp_format),
            "annotator_type": type(self.annotator).__name__,
            "annotator_prompt": self.annotator.describe(),
            "ctx_length": self.history_ctx_len,
            "logs": self.annotation_logs,
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
class LLMAnnotatorData:
    """
    A dataclass responsible for serializing and deserializing data needed to construct a
    :class:`AnnotationConv`.
    """
    attributes: list[str]
    instructions: str
    history_ctx_len: int = 4

    @staticmethod
    def from_json_file(input_file_path: str):
        """
        Construct a LLMAnnotatorData instance according to a serialized .json file.

        :param input_file_path: The path to the serialized .json file
        :type input_file_path: str
        :return: A LLMConvData instance containing the information from the file
        :rtype: LLMConvData
        """
        with open(input_file_path, "r", encoding="utf8") as fin:
            data_dict = json.load(fin)

        # code from https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary
        field_set = {f.name for f in dataclasses.fields(LLMAnnotatorData) if f.init}
        filtered_arg_dict = {k: v for k, v in data_dict.items() if k in field_set}
        return LLMAnnotatorData(**filtered_arg_dict)

    def to_json_file(self, output_path: str) -> None:
        """
        Serialize the data to a .json file.

        :param output_path: The path of the new file
        :type output_path: str
        """
        with open(output_path, "w", encoding="utf8") as fout:
            json.dump(dataclasses.asdict(self), fout, indent=4)


class LLMAnnotationGenerator:
    """
    A class responsible for creating a :class:`AnnotationConv` from the conversation data
    (:class:`LLMAnnotatorData`) and a model (:class:`lib.models.LlamaModel`).
    """

    def __init__(self,
                 data: LLMAnnotatorData,
                 llm: lib.models.LlamaModel,
                 conv_logs_path: str
                 ):
        self.data = data
        self.llm = llm
        self.conv_logs_path = conv_logs_path

    def produce_conversation(self) -> AnnotationConv:
        """
        Generate the synthetic annotations.

        :return: An initialized AnnotationConv instance.
        :rtype: AnnotationConv
        """
        annotator = lib.actors.LLMAnnotator(model=self.llm,
                                            name="",
                                            role="expert annotator",
                                            attributes=self.data.attributes,
                                            context="",
                                            instructions=self.data.instructions)

        conversation = AnnotationConv(annotator=annotator,
                                      conv_logs_path=self.conv_logs_path,
                                      history_ctx_len=self.data.history_ctx_len)
        return conversation
