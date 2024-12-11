import abc
import typing


class Model(abc.ABC):

    def __init__(self, stop_list: list[str] = []):
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
