"""
Module defining LLM users in discussions and their characteristics.
"""

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
import dataclasses
from pathlib import Path
import json
from enum import Enum, auto

from . import model
from . import _file_util


class ActorType(str, Enum):
    """
    The purpose of the LLMActor, used to determine proper prompt structure
    """

    USER = auto()
    ANNOTATOR = auto()


@dataclasses.dataclass
class Persona:
    """
    A dataclass holding information about the synthetic persona of a LLM actor.
    Includes name, Sociodemographic Background, personality
    and special instructions.
    """

    username: str = ""
    age: int = -1
    sex: str = ""
    sexual_orientation: str = ""
    demographic_group: str = ""
    current_employment: str = ""
    education_level: str = ""
    special_instructions: str = ""
    personality_characteristics: list[str] = dataclasses.field(
        default_factory=list
    )

    @classmethod
    def from_json_file(file_path: Path) -> list:
        """
        Generate a list of personas from a properly formatted persona JSON
        file.

        :param file_path: the path to the JSON file containing the personas
        :type file_path: Path
        :return: a list of LlmPersona objects for each of the file entries
        :rtype: list[LlmPersona]
        """
        all_personas = _file_util.read_json_file(file_path)

        persona_objs = []
        for data_dict in all_personas:
            # code from https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary # noqa: E501
            field_set = {f.name for f in dataclasses.fields(Persona) if f.init}
            filtered_arg_dict = {
                k: v for k, v in data_dict.items() if k in field_set
            }
            persona_obj = Persona(**filtered_arg_dict)
            persona_objs.append(persona_obj)

        return persona_objs

    def to_dict(self):
        return dataclasses.asdict(self)

    def to_json_file(self, output_path: str) -> None:
        """
        Serialize the data to a .json file.

        :param output_path: The path of the new file
        :type output_path: str
        """
        _file_util.dict_to_json(self.to_dict(), output_path)

    def __str__(self):
        return json.dumps(self.to_dict())


class Actor:
    """
    An abstract class representing an actor which responds according to an
    underlying LLM instance.
    """

    def __init__(
        self,
        model: model.BaseModel,
        persona: Persona,
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
        return self._system_prompt()["content"]

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
