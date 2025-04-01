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

import dataclasses
from pathlib import Path

from ..util import file_util


@dataclasses.dataclass
class LLMPersona:
    """
    A dataclass holding information about the synthetic persona of a LLM actor.
    Includes name, Sociodemographic Background, personality
    and special instructions.
    """

    username: str
    age: int
    sex: str
    sexual_orientation: str
    demographic_group: str
    current_employment: str
    education_level: str
    special_instructions: str
    personality_characteristics: list[str]

    def to_json_file(self, output_path: str) -> None:
        """
        Serialize the data to a .json file.

        :param output_path: The path of the new file
        :type output_path: str
        """
        file_util.dict_to_json(dataclasses.asdict(self), output_path)

    def to_attribute_list(self) -> list[str]:
        """
        Turn the various attributes of a persona into a cohesive
        list of attributes.
        """
        attributes = [
            f"{field}: {getattr(self, field)}"
            for field in dataclasses.asdict(self)
        ]
        return attributes

    @staticmethod
    def _sex_parse(sex: str) -> str:
        """
        Helper function which transforms the sex attribute of a persona into a
        prompt-friendly equivalent.
        """
        sex = sex.lower()
        if sex == "male":
            return "man"
        elif sex == "female":
            return "woman"
        else:
            return "non-binary"


def from_json_file(file_path: Path) -> list[LLMPersona]:
    """
    Generate a list of personas from a properly formatted persona JSON file.

    :param file_path: the path to the JSON file containing the personas
    :type file_path: Path
    :return: a list of LlmPersona objects for each of the file entries
    :rtype: list[LlmPersona]
    """
    all_personas = file_util.read_json_file(file_path)

    persona_objs = []
    for data_dict in all_personas:
        # code from https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary # noqa: E501
        field_set = {f.name for f in dataclasses.fields(LLMPersona) if f.init}
        filtered_arg_dict = {
            k: v for k, v in data_dict.items() if k in field_set
        }
        persona_obj = LLMPersona(**filtered_arg_dict)
        persona_objs.append(persona_obj)

    return persona_objs
