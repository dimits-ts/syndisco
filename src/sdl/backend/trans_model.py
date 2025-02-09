"""
Loads and manages HuggingFace Transformers models.
"""
import typing
import logging
from pathlib import Path

import transformers

from . import model

logger = logging.getLogger(Path(__file__).name)


class TransformersModel(model.Model):
    """
    A class encapsulating Transformers HuggingFace models.
    """
    
    def __init__(
        self,
        model_path: str | Path,
        name: str,
        max_out_tokens: int,
        remove_string_list: list[str] = [],
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
