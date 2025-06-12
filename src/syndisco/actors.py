"""
SynDisco: Automated experiment creation and execution using only LLM agents
Copyright (C) 2025 Dimitris Tsirmpas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at tsirbasdim@gmail.com
"""

import typing
from enum import Enum, auto

from . import model
from . import persona


class ActorType(str, Enum):
    """
    The purpose of the LLMActor, used to determine proper prompt structure
    """

    USER = auto()
    ANNOTATOR = auto()


class LLMActor:
    """
    An abstract class representing an actor which responds according to an
    underlying LLM instance.
    """

    def __init__(
        self,
        model: model.BaseModel,
        persona: persona.LLMPersona,
        context: str,
        instructions: str,
        actor_type: ActorType,
    ) -> None:
        """
        Create an Actor controlled by an LLM instance with a specific persona.

        :param model:
            A wrapper encapsulating a promptable LLM instance.
        :type model:
            model.BaseModel
        :param persona:
            The actor's persona.
        :type persona:
            persona.LLMPersona
        :param context:
            The context of the discussion.
        :type context:
            str
        :param instructions:
            The actor instructions for the discussion.
        :type instructions:
            str
        :param actor_type:
            Whether the actor is an annotator or participant.
        :type actor_type:
            ActorType
        """
        self.model = model
        self.persona = persona
        self.context = context
        self.instructions = instructions
        self.actor_type = actor_type

    def _system_prompt(self) -> dict:
        prompt = {
            "context": self.context,
            "instructions": self.instructions,
            "type": self.actor_type,
            "persona": self.persona.to_dict(),
        }
        return {"role": "system", "content": prompt}

    def _message_prompt(self, history: list[str]) -> dict:
        return _apply_template(self.actor_type, self.get_name(), history)

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
        response = self.model.prompt(
            (system_prompt, message_prompt),
            stop_words=["###", "\n\n", "User"],
        )
        return response

    def describe(self) -> dict:
        """
        Get a description of the actor's internals.

        :return: A brief description of the actor
        :rtype: dict
        """
        return self._system_prompt()['content']

    @typing.final
    def get_name(self) -> str:
        """
        Get the actor's assigned name within the conversation.

        :return: The name of the actor.
        :rtype: str
        """
        return self.persona.username


def _apply_template(
    actor_type: ActorType, username: str, history: list[str]
) -> dict[str, str]:
    if actor_type == ActorType.USER:
        return {
            "role": "user",
            "content": "\n".join(history) + f"\nUser {username} posted:",
        }
    elif actor_type == ActorType.ANNOTATOR:
        # LLMActor asks the model to respond as its username
        # by modifying this protected method, we instead prompt
        # it to write the annotation
        return {
            "role": "user",
            "content": "Conversation so far:\n\n"
            + "\n".join(history)
            + "\nOutput:",
        }
