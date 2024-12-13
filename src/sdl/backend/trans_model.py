import transformers

import typing

from . import model


class TransformersModel(model.Model):
    def __init__(
        self,
        model_path: str,
        name: str,
        max_out_tokens: int,
        remove_string_list=[],
        device: int = -1,
    ):
        """
        Initialize a new LLM wrapper.

        :param model_path: the full path to the GGUF model file e.g. 'openai-community/gpt2'
        :type model: str
        :param name: your own name for the model e.g. 'GPT-2'
        :type name: str
        :param max_out_tokens: the maximum number of tokens in the response
        :type max_out_tokens: int
        :param remove_string_list: a list of strings to be removed from the response.
        Used to prevent model-specific conversational collapse, defaults to []
        :type remove_string_list: list, optional
        """
        super().__init__(name, max_out_tokens, remove_string_list)

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

        self.generator = transformers.pipeline(
            "text-generation", model=model, tokenizer=tokenizer
        )

    def generate_response(
        self, json_prompt: tuple[typing.Any, typing.Any], stop_words: list[str]
    ) -> str:
        return self.generator(json_prompt, max_length=self.max_out_tokens, return_full_text=False)  # type: ignore
