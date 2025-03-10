
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


"""
Module containing wrappers for local LLMs loaded with various Python libraries.
"""

import abc
import typing
import logging
from pathlib import Path

import transformers


logger = logging.getLogger(Path(__file__).name)


class BaseModel(abc.ABC):
    """
    Interface for all local LLM wrappers
    """

    def __init__(self, name: str, max_out_tokens: int, stop_list: list[str] | None = None):
        self.name = name
        self.max_out_tokens = max_out_tokens
        # avoid mutable default value problem
        self.stop_list = stop_list if stop_list is not None else []

    @typing.final
    def prompt(
        self,
        json_prompt: tuple[typing.Any, typing.Any],
        stop_words: list[str]
    ) -> str:
        """Generate the model's response based on a prompt.

        :param json_prompt: A tuple containing the system and user prompt. Could be strings, or a dictionary.
        :type json_prompt: tuple[typing.Any, typing.Any]
        :param stop_words: Strings where the model should stop generating
        :type stop_words: list[str]
        :return: the model's response
        :rtype: str
        """
        response = self.generate_response(json_prompt, stop_words)
        # avoid model collapse attributed to certain strings
        for remove_word in self.stop_list:
            response = response.replace(remove_word, "")

        return response

    @abc.abstractmethod
    def generate_response(self,
        json_prompt: tuple[typing.Any, typing.Any],
        stop_words) -> str:
        """Model-specific method which generates the LLM's response

        :param json_prompt: A tuple containing the system and user prompt. Could be strings, or a dictionary.
        :type json_prompt: tuple[typing.Any, typing.Any]
        :param stop_words: Strings where the model should stop generating
        :type stop_words: list[str]
        :return: the model's response
        :rtype: str
        """
        raise NotImplementedError("Abstract class call")


class TransformersModel(BaseModel):
    """
    A class encapsulating Transformers HuggingFace models.
    """
    
    def __init__(
        self,
        model_path: str | Path,
        name: str,
        max_out_tokens: int,
        remove_string_list: list[str] | None = None,
    ):
        """
        Initialize a new LLM wrapper.

        :param model_path: the full path to the GGUF model file e.g. 'openai-community/gpt2'
        :param name: your own name for the model e.g. 'GPT-2'
        :param max_out_tokens: the maximum number of tokens in the response
        :param remove_string_list: a list of strings to be removed from the response.
        """
        super().__init__(name, max_out_tokens, remove_string_list)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto"
        )

        logger.info(
            f"Model memory footprint:  {self.model.get_memory_footprint()/2**20:.2f} MBs"
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        self.generator = transformers.pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer
        )

    def generate_response(
        self, json_prompt: tuple[typing.Any, typing.Any], stop_words: list[str]
    ) -> str:
        """
        Generate a response using the model's chat template.

        :param chat_prompt: A list of dictionaries representing the chat history.
        :param stop_words: A list of stop words to prevent overflow in responses.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                json_prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            logger.warning(
                "No chat template found in model's tokenizer: Falling back to default..."
            )
            formatted_prompt = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in json_prompt
            )

        response = self.generator(
            formatted_prompt, max_new_tokens=self.max_out_tokens, return_full_text=False
        )[0]["generated_text"]  # type: ignore

        return response  # type: ignore
