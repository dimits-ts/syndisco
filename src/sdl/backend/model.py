"""
Module containing wrappers for local LLMs loaded with various Python libraries.
"""

import abc
import typing
import logging
from pathlib import Path

import llama_cpp
import transformers


logger = logging.getLogger(Path(__file__).name)


class BaseModel(abc.ABC):
    """
    Interface for all local LLM wrappers
    """

    def __init__(self, name: str, max_out_tokens: int, stop_list: list[str] = list()):
        self.name = name
        self.max_out_tokens = max_out_tokens
        self.stop_list = stop_list

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


class LlamaModel(BaseModel):
    """
    Wrapper for local LLMs loaded via the llama.cpp library.
    Uses llama-cpp-python to manage the models
    """

    def __init__(
        self,
        model_path: Path,
        name: str,
        gpu_layers: int,
        seed: int = 42,
        ctx_width_tokens: int = 2048,
        max_out_tokens: int = 400,
        inference_threads: int = 3,
        remove_string_list: list[str] = list(),
    ):
        """
        Initialize a new LLM wrapper.

        :param model_path: the LLM to be used
        :type model_path: llama_cpp.Llama
        :param name: a shorthand name for the model used
        :type name: str
        :param max_out_tokens: the maximum number of tokens in the response
        :type max_out_tokens: int
        :param seed: random seed
        :type seed: int
        :param ctx_width_tokens: the number of tokens available for context
        :type ctx_width_tokens: int
        :param inference_threads: how many CPU threads will run on the RAM-allocated tensors
        :type inference_threads: int
        :param gpu_layers: how many layers will be offloaded to the GPU
        :type gpu_layers: int
        :param remove_string_list: a list of strings to be removed from the response.
        Used to prevent model-specific conversational collapse, defaults to []
        :type remove_string_list: list, optional
        """
        super().__init__(name, max_out_tokens, remove_string_list)

        self.model = llama_cpp.Llama(
            model_path=str(model_path),
            seed=seed,
            n_ctx=ctx_width_tokens,
            n_threads=inference_threads,
            n_gpu_layers=gpu_layers,
            use_mmap=True,
            chat_format="alpaca",
            mlock=True,
            verbose=False,
        )
        self.max_out_tokens = max_out_tokens
        self.seed = seed

    def generate_response(
        self, json_prompt: tuple[typing.Any, typing.Any], stop_words: list[str]
    ) -> str:
        output = self.model.create_chat_completion(
            messages=json_prompt,  # type: ignore
            max_tokens=self.max_out_tokens,
            seed=self.seed,
            stop=stop_words,
        )  # prevent model from generating the next actor's response

        response = self._get_response_from_output(output)

        return response

    @staticmethod
    def _get_response_from_output(json_output) -> str:
        """
        Extracts the model's response from the raw output as a string.
        """
        return json_output["choices"][0]["message"]["content"]


class TransformersModel(BaseModel):
    """
    A class encapsulating Transformers HuggingFace models.
    """
    
    def __init__(
        self,
        model_path: str | Path,
        name: str,
        max_out_tokens: int,
        remove_string_list: list[str] = list(),
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
