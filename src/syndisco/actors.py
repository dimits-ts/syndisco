"""
Module defining LLM users in discussions and their characteristics.
"""

# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2025 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr


import typing
import json

from . import model


class Actor:
    """
    An abstract class representing an actor which responds according to an
    underlying LLM instance.
    """

    def __init__(
        self,
        model: model.BaseModel,
        persona: dict[str, str],
        context: str,
        instructions: str,
        is_annotator: bool = False,
        name: str = "<Unnamed>"
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
            dict[str, str]
        :param context:
            The context of the discussion.
        :type context:
            str
        :param instructions:
            The actor instructions for the discussion.
        :type instructions:
            str
        :param is_annotator:
            Whether the actor is an annotator or discussion participant.
        :type actor_type:
            bool
        """
        self.model = model
        self.persona = persona
        self.context = context
        self.instructions = instructions
        self.is_annotator = is_annotator
        self.name = name

    def get_system_prompt(self) -> str:
        """
        Get the system prompt provided to the agent.
        :return: The system prompt provided to the agent.
        :rtype: str
        """
        prompt = {
            "context": self.context,
            "instructions": self.instructions,
            "type": "annotator" if self.is_annotator else "user",
            "persona": {
                item[0]: item[1]
                for item in self.persona.items()
            },
        }
        return json.dumps(prompt)

    def get_user_prompt(self, history: list[str] | None = None) -> str:
        """
        Get the message prompt provided to the agent.
        Changes depending on whether the agent is an annotator or a user.

        :param history:
            The history of previous messages. Each element in the list
            corresponds to one message, including relevant information
            (such as the name of the user who posted it).
            None if no discussion history exists.
        :type history: list[str] | None
        :return: The message prompt provided to the agent.
        :rtype: str
        """
        history_str = (
            f"Conversation so far:\n{"\n".join(history)}"
            if history is not None
            else ""
        )
        if self.is_annotator:
            # LLMActor asks the model to respond as its username
            # we instead prompt it to write the annotation
            json_input = {
                "role": "user",  # do not confuse this with our own roles
                "content": history_str + "\nOutput:",
            }
        else:
            json_input = {
                "role": "user",
                "content": (
                    history_str + f"\nUser {self.name} posted:"
                ),
            }

        return json.dumps(json_input)

    @typing.final
    def speak(self, history: list[str] | None = None) -> str:
        """
        Prompt the actor to speak, given a history of previous messages
        in the conversation (None if no history).

        This method should not be modified. If you are subclassing Actor,
        modify the :meth:get_user_prompt and :meth:get_system_prompt methods
        instead.

        :param history: A list of previous messages.
        :type history: list[str]
        :return: The actor's new message
        :rtype: str
        """
        system_prompt = self.get_system_prompt()
        message_prompt = self.get_user_prompt(history)
        response = self.model.prompt(system_prompt, message_prompt)
        return response

    @typing.final
    def get_name(self) -> str:
        """
        Get the actor's assigned name within the conversation.

        :return: The name of the actor.
        :rtype: str
        """
        return self.name
