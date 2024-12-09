import abc
import llama_cpp


class Model(abc.ABC):

    @abc.abstractmethod
    def prompt(
        self,
        json_prompt: list[llama_cpp.ChatCompletionRequestMessage],
        stop_list: list[str],
    ) -> str:
        raise NotImplementedError


class LlamaModel(Model):

    @staticmethod
    def _get_response_from_output(json_output) -> str:
        """
        Extracts the model's response from the raw output as a string.
        """
        return json_output["choices"][0]["message"]["content"]

    def __init__(
        self,
        model: llama_cpp.Llama,
        name: str,
        max_out_tokens: int,
        seed: int,
        remove_string_list=[],
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
        self.model = model
        self.max_out_tokens = max_out_tokens
        self.seed = seed
        self.remove_string_list = remove_string_list
        self.name = name

    def prompt(
        self,
        json_prompt: list[llama_cpp.ChatCompletionRequestMessage],
        stop_list: list[str],
    ) -> str:
        output = self.model.create_chat_completion(
            messages=json_prompt,
            max_tokens=self.max_out_tokens,
            seed=self.seed,
            stop=["###", "\n\n"] + stop_list,
        )  # prevent model from generating the next actor's response

        response = self._get_response_from_output(output)
        
        # model collapse including this string present in llama3
        for remove_word in self.remove_string_list:
            response = response.replace(remove_word, "")

        return response
