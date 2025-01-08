import json
import collections
import datetime
import textwrap
from typing import Any
from pathlib import Path

from ..backend import actors
from ..util import output_util, file_util


# Compared to conversations.py one may think that this violates DRY
# HOwever, for the most part the API is the same by convention, not because it
# uses the same functionality. You can replace the implementation here entirely
# without impacting conversations.py at all


class AnnotationConv:
    """
    An annotation job modelled as a conversation between the messages of a finished dialogue, and the LLM Annotator.
    """

    def __init__(
        self,
        annotator: actors.LLMAnnotator,
        conv_logs_path: str | Path,
        include_moderator_comments: bool,
        history_ctx_len: int = 2
    ):
        """Create an annotation job.
        The annotation is modelled as a conversation between the system and the annotator.

        :param annotator: The annotator
        :type annotator: actors.IActor
        :param conv_logs_path: The path to the file containing the conversation logs in JSON format
        :type conv_logs_path: str | Path
        :param include_moderator_comments: Whether to annotate moderator comments, and include them in conversational context when annotating user responses.
        :type include_moderator_comments: bool
        :param history_ctx_len: How many previous comments the annotator will remember, defaults to 4
        :type history_ctx_len: int, optional
        """
        self.annotator = annotator
        self.history_ctx_len = history_ctx_len
        self.include_moderator_comments = include_moderator_comments
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

        for message_data in self.conv_data_dict["logs"]:
            username = message_data["name"]
            message = message_data["text"]

            # do not include moderator comments in annotation context if told so
            if "moderator" in username:
                if not self.include_moderator_comments:
                    continue
            formatted_message = output_util.format_chat_message(username, message)
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

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)
