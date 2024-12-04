import abc
import typing

from ..backend import models


class LlmActor(abc.ABC):
    """
    An abstract class representing an actor which responds according to an underlying LLM instance.
    The LLM instance can be of any type, provided it satisfies the 
    :class:`models.IGeneratingAgent` interface.
    """

    def __init__(self,
                 model: models.LlamaModel,
                 name: str,
                 attributes: list[str],
                 context: str,
                 instructions: str) -> None:
        """
        Create a new actor based on an LLM instance.

        :param model: A model or wrapper encapsulating a promptable LLM instance.
        :type model: tasks.models.LlamaModel
        :param name: The name given to the in-conversation actor.
        :type name: str
        :type role: str
        :param attributes: A list of attributes which characterize the actor
         (e.g. "middle-class", "LGBTQ", "well-mannered").
        :type attributes: list[str]
        :param context: The context of the conversation, including topic and participants.
        :type context: str
        :param instructions: Special instructions for the actor.
        :type instructions: str
        """
        self.model = model
        self.name = name
        self.attributes = attributes
        self.context = context
        self.instructions = instructions

    def _system_prompt(self) -> dict:
        prompt = f"{self.context} Your name is {self.name}. Your traits: {", ".join(self.attributes)} Your instructions: {self.instructions}"
        return {"role": "system", "content": prompt}

    @abc.abstractmethod
    def _message_prompt(self, history: list[str]) -> dict:
        return {}

    @typing.final
    def speak(self, history: list[str]) -> str:
        """
        Prompt the actor to speak, given a history of previous messages
        in the conversation.

        :param history: A list of previous messages.
        :type history: list[str]
        :return: The actor's new message
        :rtype: str
        """
        system_prompt = self._system_prompt()
        message_prompt = self._message_prompt(history)
        # debug
        print("System prompt: ", system_prompt)
        print("Message prompt: ", message_prompt)
        print("Response:")
        response = self.model.prompt([system_prompt, message_prompt], stop_list=["User"]) #type: ignore
        return response

    def describe(self):
        """
        Get a description of the actor's internals.

        :return: A brief description of the actor
        :rtype: str
        """
        return f"{self._system_prompt()["content"]}"

    @typing.final
    def get_name(self) -> str:
        """
        Get the actor's assigned name within the conversation.

        :return: The name of the actor.
        :rtype: str
        """
        return self.name


class LLMUser(LlmActor):
    """
    A LLM actor with a modified message prompt to facilitate a conversation.
    """
    def _message_prompt(self, history: list[str]) -> dict:
        return {
            "role": "user",
            "content": "\n".join(history) + f"\nUser {self.get_name()} posted:"
        }

class LLMAnnotator(LlmActor):
    """
    A LLM actor with a modified message prompt to facilitate an annotation job.
    """

    def _message_prompt(self, history: list[str]) -> dict:
        # LLMActor asks the model to respond as its username
        # by modifying this protected method, we instead prompt it to write the annotation
        return {
            "role": "user",
            "content": "Conversation so far:\n\n" + "\n".join(history) + "\nOutput:"
        }
