import abc
import typing

import lib.models


class IActor(abc.ABC):
    """
    An interface denoting an actor within a conversation.
    Designed to be as general as possible, in order to support LLM, human and
    IR/random selection models.
    """

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the actor's assigned name within the conversation.

        :return: The name of the actor.
        :rtype: str
        """
        return ""

    @abc.abstractmethod
    def speak(self, history: list[str]) -> str:
        """
        Prompt the actor to speak, given a history of previous messages
        in the conversation.

        :param history: A list of previous messages.
        :type history: list[str]
        :return: The actor's new message
        :rtype: str
        """
        return ""

    @abc.abstractmethod
    def describe(self) -> str:
        """
        Get a description of the actor's internals.

        :return: A brief description of the actor
        :rtype: str
        """
        return ""


class ALlmActor(IActor, abc.ABC):
    """
    An abstract class representing an actor which responds according to an underlying LLM instance.
    The LLM instance can be of any type, provided it satisfies the 
    :class:`lib.models.IGeneratingAgent` interface.
    """

    def __init__(self,
                 model: lib.models.LlamaModel,
                 name: str,
                 role: str,
                 attributes: list[str],
                 context: str,
                 instructions: str) -> None:
        """
        Create a new actor based on an LLM instance.

        :param model: A model or wrapper encapsulating a promptable LLM instance.
        :type model: tasks.models.LlamaModel
        :param name: The name given to the in-conversation actor.
        :type name: str
        :param role: The role of the actor within the conversation 
        (e.g. "chat user", "chat moderator").
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
        self.role = role
        self.attributes = attributes
        self.context = context
        self.instructions = instructions

    def _system_prompt(self) -> dict:
        prompt = f"You are {self.name} a {", ".join(self.attributes)} {self.role}. {self.context} {self.instructions}."
        return {"role": "system", "content": prompt}

    @abc.abstractmethod
    def _message_prompt(self, history: list[str]) -> dict:
        return {}

    def describe(self):
        return f"Model: {type(self.model).__name__}. Prompt: {self._system_prompt()["content"]}"

    @typing.final
    def speak(self, history: list[str]) -> str:
        system_prompt = self._system_prompt()
        message_prompt = self._message_prompt(history)
        response = self.model.prompt([system_prompt, message_prompt], stop_list=["User"])
        return response

    @typing.final
    def get_name(self) -> str:
        return self.name


class LLMUser(ALlmActor):
    """
    A LLM actor with a modified message prompt to facilitate a conversation.
    """
    def _message_prompt(self, history: list[str]) -> dict:
        return {
            "role": "user",
            "content": "\n".join(history) + f"\n{self.get_name()}:"
        }

class LLMAnnotator(ALlmActor):
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
