"""
Module containing wrappers for local LLMs loaded with various Python libraries.
"""

# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2025 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr
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

    def __init__(
        self,
        name: str,
        max_out_tokens: int,
        stop_list: list[str] | None = None,
    ):
        self.name = name
        self.max_out_tokens = max_out_tokens
        # avoid mutable default value problem
        self.stop_list = stop_list if stop_list is not None else []

    @typing.final
    def prompt(self, json_prompt: tuple[typing.Any, typing.Any]) -> str:
        """Generate the model's response based on a prompt.

        :param json_prompt:
            A tuple containing the system and user prompt.
            Could be strings, or a dictionary.
        :type json_prompt: tuple[typing.Any, typing.Any]
        :param stop_words: Strings where the model should stop generating
        :type stop_words: list[str]
        :return: the model's response
        :rtype: str
        """
        response = self.generate_response(json_prompt)
        # avoid model collapse attributed to certain strings
        for remove_word in self.stop_list:
            response = response.replace(remove_word, "")

        return response

    @abc.abstractmethod
    def generate_response(
        self,
        json_prompt: tuple[typing.Any, typing.Any],
    ) -> str:
        """Model-specific method which generates the LLM's response

        :param json_prompt:
            A tuple containing the system and user prompt.
            Could be strings, or a dictionary.
        :type json_prompt:
            tuple[typing.Any, typing.Any]
        :return:
            The model's response
        :rtype: str
        """
        raise NotImplementedError("Abstract class call")

    @typing.final
    def get_name(self) -> str:
        """
        Get the model's assigned pseudoname.

        :return: The name of the model.
        :rtype: str
        """
        return self.name


class TransformersModel(BaseModel):
    """
    HuggingFace Transformers model wrapper.
    """

    def __init__(
        self,
        model_path: str | Path,
        name: str,
        max_out_tokens: int,
        remove_string_list: list[str] | None = None,
    ):
        super().__init__(name, max_out_tokens, remove_string_list)

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        model_size = self.model.get_memory_footprint() / 2**20
        logger.info(f"Model memory footprint: {model_size:.2f} MB")

    def generate_response(self, json_prompt: tuple[str, str]) -> str:

        system_prompt, user_prompt = json_prompt

        # Construct proper message list for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prefer chat template if available
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            logger.warning("Tokenizer has no chat template; falling back.")
            prompt_text = (
                f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"
            )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
            self.model.device
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_out_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Remove the prompt portion → keep only generated part
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return response.strip()
