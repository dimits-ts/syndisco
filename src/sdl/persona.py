import dataclasses
import json


@dataclasses.dataclass
class LlmPersona:

    name: str
    age: int
    sex: str
    sexual_orientation: str
    demographic_group: str
    current_employment: str
    education_level: str
    intent: str
    personality_characteristics: list[str]

    @staticmethod
    def from_json_file(file_path: str):
        with open(file_path, "r", encoding="utf-8") as file:
            data_dict = json.load(file)

        # code from https://stackoverflow.com/questions/68417319/initialize-python-dataclass-from-dictionary
        field_set = {f.name for f in dataclasses.fields(LlmPersona) if f.init}
        filtered_arg_dict = {k: v for k, v in data_dict.items() if k in field_set}
        return LlmPersona(**filtered_arg_dict)

    def to_json_file(self, output_path: str) -> None:
        """
        Serialize the data to a .json file.

        :param output_path: The path of the new file
        :type output_path: str
        """
        with open(output_path, "w+", encoding="utf8") as fout:
            json.dump(dataclasses.asdict(self), fout, indent=4)
   

    def to_attribute_list(self) -> list[str]:
        """Turn the various attributes of a persona into a cohesive list of attributes to be used in the model prompt.

        :return: a list of attributes to be given as input for a model prompt
        :rtype: list[str]
        """
        attributes = []
        attributes.append(f"{self.age} years old")
        attributes.append(self.sexual_orientation)
        attributes.append(self.demographic_group)
        attributes.append(self.current_employment)

        for characteristic in self.personality_characteristics:
            attributes.append(characteristic)

        attributes.append(LlmPersona._sex_parse(self.sex))
        attributes.append(f"with {self.education_level} education")
        attributes.append(f"and {self.intent} intent")

        return attributes

    @staticmethod
    def _sex_parse(sex: str) -> str:
        """Helper function which transforms the sex attribute of a persona into a prompt-friendly equivalent."""
        match sex.lower():
            case "male":
                return "man"
            case "female":
                return "woman"
            case _:
                return "non-binary"
