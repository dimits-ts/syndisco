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
        device: int = -1
    ):
        """
        Initialize a new LLM wrapper.

        :param model_path: the path to the GGUF model file
        :type model: str
        :param name: the transformers name of the model e.g.'openai-community/gpt2'
        :type name: str
        :param max_out_tokens: the maximum number of tokens in the response
        :type max_out_tokens: int
        :param remove_string_list: a list of strings to be removed from the response. 
        Used to prevent model-specific conversational collapse, defaults to []
        :type remove_string_list: list, optional
        """
        super().__init__(remove_string_list)
        self.max_out_tokens = max_out_tokens
        self.remove_string_list = remove_string_list
        self.name = name
        
        model = transformers.AutoModelForCausalLM.from_pretrained(name, gguf_file=model_path, device_map = 'auto')
        self.generator = transformers.pipeline("text-generation", model=model)
        

    def generate_response(
        self,
        json_prompt: tuple[typing.Any, typing.Any],
        stop_words: list[str]
    ) -> str:
        return self.generator(json_prompt, max_length=self.max_out_tokens)