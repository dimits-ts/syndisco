from . import annotation
from . import models
from . import actors

import dataclasses
import json


@dataclasses.dataclass
class LlmAnnotationData:
    """
    A dataclass responsible for serializing and deserializing data needed to construct a
    :class:`AnnotationConv`.
    """

    attributes: list[str]
    instructions: str
    include_moderator_comments: bool
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
        field_set = {f.name for f in dataclasses.fields(LlmAnnotationData) if f.init}
        filtered_arg_dict = {k: v for k, v in data_dict.items() if k in field_set}
        return LlmAnnotationData(**filtered_arg_dict)

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
    (:class:`LLMAnnotatorData`) and a model (:class:`models.LlamaModel`).
    """

    def __init__(
        self, data: LlmAnnotationData, llm: models.LlamaModel, conv_logs_path: str
    ):
        assert data is not None and llm is not None and conv_logs_path is not None
        self.data = data
        self.llm = llm
        self.conv_logs_path = conv_logs_path

    def produce_conversation(self) -> annotation.AnnotationConv:
        """
        Generate the synthetic annotations.

        :return: An initialized AnnotationConv instance.
        :rtype: AnnotationConv
        """
        annotator = actors.LLMAnnotator(
            model=self.llm,
            name="",
            attributes=self.data.attributes,
            context="",
            instructions=self.data.instructions,
        )

        conversation = annotation.AnnotationConv(
            annotator=annotator,
            conv_logs_path=self.conv_logs_path,
            history_ctx_len=self.data.history_ctx_len,
            include_moderator_comments=self.data.include_moderator_comments,
        )
        return conversation
