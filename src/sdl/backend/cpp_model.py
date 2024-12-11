import llama_cpp
import typing

from . import model


class LlamaModel(model.Model):

    def __init__(
        self,
        model: llama_cpp.Llama,
        name: str,
        max_out_tokens: int,
        seed: int,
        remove_string_list: list[str] = []
    ):
        """
        Initialize a new LLM wrapper.

        :param model: the LLM to be used
        :type model: llama_cpp.Llama
        :param name: a shorthand name for the model used
        :type name: str
        :param max_out_tokens: the maximum number of tokens in the response
        :type max_out_tokens: int
        :param seed: random seed
        :type seed: int
        :param remove_string_list: a list of strings to be removed from the response. 
        Used to prevent model-specific conversational collapse, defaults to []
        :type remove_string_list: list, optional
        """
        super().__init__(remove_string_list)
        self.model = model
        self.max_out_tokens = max_out_tokens
        self.seed = seed
        self.name = name

    def generate_response(
        self,
        json_prompt: tuple[typing.Any, typing.Any],
        stop_words: list[str]
    ) -> str:
        output = self.model.create_chat_completion(
            messages=json_prompt, # type: ignore
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