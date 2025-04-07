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
from pathlib import Path
from enum import Enum, auto

from . import model
from . import persona
from ..util import file_util


class ActorType(Enum):
    """
    The purpose of the LLMActor, used to determine proper prompt structure
    """

    USER = auto()
    ANNOTATOR = auto()


class LLMActor:
    """
    An abstract class representing an actor which responds according to an underlying LLM instance.
    The LLM instance can be of any type.
    """

    def __init__(
        self,
        model: model.BaseModel,
        name: str,
        attributes: list[str],
        context: str,
        instructions: str,
        actor_type: ActorType,
    ) -> None:
        """
        Create a new actor based on an LLM instance.

        :param model: A model or wrapper encapsulating a promptable LLM instance.
        :type model: tasks.cpp_model.LlamaModel
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
        :param actor_type: The purpose of the actor
        :type actor_type: ActorType
        """
        self.model = model
        self.name = name
        self.attributes = attributes
        self.context = context
        self.instructions = instructions
        self.actor_type = actor_type

    def _system_prompt(self) -> dict:
        prompt = f"{self.context} Your name is {self.name}. Your traits: {', '.join(self.attributes)} " + \
        f"Your instructions: {self.instructions}"
        return {"role": "system", "content": prompt}

    def _message_prompt(self, history: list[str]) -> dict:
        return _apply_template(self.actor_type, self.name, history)

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
        # print("System prompt: ", system_prompt)
        # print("Message prompt: ", message_prompt)
        # print("Response:")
        response = self.model.prompt(
            (system_prompt, message_prompt), stop_words=["###", "\n\n", "User"]
        )
        return response

    def describe(self):
        """
        Get a description of the actor's internals.

        :return: A brief description of the actor
        :rtype: str
        """
        return f"{self._system_prompt()['content']}"

    @typing.final
    def get_name(self) -> str:
        """
        Get the actor's assigned name within the conversation.

        :return: The name of the actor.
        :rtype: str
        """
        return self.name


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
        # by modifying this protected method, we instead prompt it to write the annotation
        return {
            "role": "user",
            "content": "Conversation so far:\n\n" + "\n".join(history) + "\nOutput:",
        }


def create_users_from_file(
    llm: model.BaseModel,
    persona_path: Path,
    instruction_path: Path,
    context: str,
    actor_type: ActorType,
) -> list[LLMActor]:
    """
    Create a list of users by using information from files.

    :param llm: The LLM
    :type llm: model.BaseModel
    :param persona_path: The path to the JSON file containing the personas
    :type persona_path: Path
    :param instruction_path: The path to the file containing the user's instructions
    :type instruction_path: Path
    :param context: The context of the experiment
    :type context: str
    :return: A list of initialized LLMActors
    :rtype: list[LLMActor]
    """
    personas = persona.from_json_file(persona_path)
    instructions = file_util.read_file(instruction_path)
    return create_users(
        llm,
        [persona.username for persona in personas],
        [persona.to_attribute_list() for persona in personas],
        context,
        instructions,
        actor_type,
    )


def create_users(
    llm: model.BaseModel,
    usernames: list[str],
    attributes: list[list[str]],
    context: str,
    instructions: str,
    actor_type: ActorType,
) -> list[LLMActor]:
    """Create runtime LLMActor objects with the specified information.

    :param llm: The LLM
    :type llm: model.BaseModel
    :param usernames: A list of usernames for each of the users
    :type usernames: list[str]
    :param attributes: A list containing a list of personality/mood attributes for each user
    :type attributes: list[list[str]]
    :param context: The context of the experiment
    :type context: str
    :param instructions: The instructions given to all LLM users (not the moderator)
    :type instructions: str
    :return: A list of initialized LLMActors
    :rtype: list[LLMActor]
    """
    user_list = []

    assert len(usernames) == len(
        attributes
    ), "Number of usernames and user personality attribute lists must be the same"

    for username, user_attributes in zip(usernames, attributes):
        user_list.append(
            LLMActor(
                model=llm,
                name=username,
                attributes=user_attributes,
                context=context,
                instructions=instructions,
                actor_type=actor_type,
            )
        )
    return user_list
