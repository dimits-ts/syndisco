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
"""
Module containing wrappers for local LLMs loaded with various Python libraries.
"""

import abc
import typing
import logging
from pathlib import Path

import transformers
import torch
from openai import OpenAI


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
    def prompt(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Generate the model's response based on a prompt.

        :param system_prompt: The system prompt.
        :type system_prompt: str
        :param user_prompt: The user prompt.
        :type user_prompt: str
        :param stop_words: Strings to be removed after generation.
        :type stop_words: list[str]
        :return: the model's response
        :rtype: str
        """
        response = self._generate_response(system_prompt, user_prompt)
        # avoid model collapse attributed to certain strings
        for remove_word in self.stop_list:
            response = response.replace(remove_word, "")

        return response

    @typing.final
    def get_name(self) -> str:
        """
        Get the model's assigned pseudoname.

        :return: The name of the model.
        :rtype: str
        """
        return self.name

    @abc.abstractmethod
    def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Model-specific method which generates the LLM's response

        :param system_prompt: The system prompt.
        :type system_prompt: str
        :param user_prompt: The user prompt.
        :type user_prompt: str
        :return: The model's response
        :rtype: str
        """
        raise NotImplementedError("Abstract class call")


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
        model_kwargs: dict | None = None,
        tokenizer_kwargs: dict | None = None,
        generation_kwargs: dict | None = None,
    ):
        """
        Initialize a HuggingFace Transformers-based language model wrapper.

        This class loads a causal language model and tokenizer from a
        HuggingFace-compatible checkpoint and configures them for inference.
        It also supports optional customization of model loading, tokenizer
        behavior, and generation parameters via keyword argument dictionaries.

        :param model_path:
            Path or HuggingFace model identifier used to load the 
            model and tokenizer.
        :type model_path: str | Path

        :param name:
            A human-readable name or alias for this model instance.
        :type name: str

        :param max_out_tokens:
            Maximum number of tokens to generate per response.
        :type max_out_tokens: int

        :param remove_string_list:
            Optional list of substrings to remove from generated outputs.
        :type remove_string_list: list[str] | None

        :param model_kwargs:
            Additional keyword arguments forwarded to
            ``transformers.AutoModelForCausalLM.from_pretrained``.
            Examples include dtype configuration, quantization settings, etc.
        :type model_kwargs: dict | None

        :param tokenizer_kwargs:
            Additional keyword arguments forwarded to
            ``transformers.AutoTokenizer.from_pretrained``.
            Examples include ``use_fast=True`` or special token configuration.
        :type tokenizer_kwargs: dict | None

        :param generation_kwargs:
            Additional keyword arguments forwarded to
            ``model.generate()`` during inference.
            Examples include sampling parameters such as ``temperature``,
            ``top_p``, or ``repetition_penalty``.
        :type generation_kwargs: dict | None

        :raises OSError:
        If the model or tokenizer cannot be loaded from the given path.

        :raises ValueError:
        If the provided configuration is invalid / incompatible with the model.

        :note:
        The model is moved to the appropriate device automatically using
        ``device_map="auto"`` and set to evaluation mode.
        """
        super().__init__(name, max_out_tokens, remove_string_list)

        model_kwargs = model_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        self.generation_kwargs = generation_kwargs or {}

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            **model_kwargs,
        ).eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            **tokenizer_kwargs,
        )

        model_size = self.model.get_memory_footprint() / 2**20
        logger.info(f"Model memory footprint: {model_size:.2f} MB")

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
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

        with torch.inference_mode():
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(
                self.model.device
            )

            output_ids = self.model.generate(  # type: ignore
                **inputs,
                max_new_tokens=self.max_out_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.generation_kwargs,
            )

        # Remove the prompt portion, keep only generated part
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        return response.strip()


class OpenAIModel(BaseModel):
    """
    OpenAI API-compatible model wrapper.

    Supports OpenAI API and compatible endpoints.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        name: str,
        max_out_tokens: int,
        temperature: float = 0.0,
        remove_string_list: list[str] | None = None,
    ):
        """Initialize the OpenAI model wrapper.

        :param model_name:
            The model identifier to use (e.g., "gpt-4", "gpt-3.5-turbo")
        :type model_name: str
        :param api_key: The API key for authentication
        :type api_key: str
        :param base_url: The base URL for the API endpoint
        :type base_url: str
        :param name: The pseudoname for this model instance
        :type name: str
        :param max_out_tokens: Maximum number of tokens to generate
        :type max_out_tokens: int
        :param temperature: Sampling temperature (0.0 for deterministic)
        :type temperature: float
        :param remove_string_list: Strings to remove from responses
        :type remove_string_list: list[str] | None
        """
        super().__init__(name, max_out_tokens, remove_string_list)

        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info(f"Initialized OpenAI model: {model_name} at {base_url}")

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using the OpenAI API.

        :param system_prompt: The system prompt.
        :type system_prompt: str
        :param user_prompt: The user prompt.
        :type user_prompt: str
        :return: The model's response
        :rtype: str
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,  # type: ignore
            max_tokens=self.max_out_tokens,
            temperature=self.temperature,
        )
        response = self._validate_response(response)
        return response

    def _validate_response(self, response: typing.Any) -> str:
        # Validate response object
        if response is None:
            raise ValueError("Received empty response from OpenAI API")

        if not hasattr(response, "choices") or not response.choices:
            raise ValueError(
                "Malformed response: missing or empty 'choices'."
                f" Full response: {response}"
            )

        choice = response.choices[0]

        if not hasattr(choice, "message") or choice.message is None:
            raise ValueError(
                "Malformed response: missing 'message' in first choice. "
                f"Choice: {choice}"
            )

        message = choice.message

        if not hasattr(message, "content"):
            raise ValueError(
                "Malformed response: missing 'content' field in message. "
                f" Message: {message}"
            )

        content = message.content

        if content is None:
            raise ValueError(
                "Model returned a response with null content "
                "(possible tool call or empty output)"
            )

        if not isinstance(content, str):
            raise TypeError(
                f"Unexpected content type: expected str, got {type(content)}"
            )

        content = content.strip()

        if not content:
            raise ValueError("Model returned empty response")

        return content
