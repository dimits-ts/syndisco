from unittest import TestCase
from lib.conversation import LLMConvData, LLMConvGenerator

import llama_cpp


class TestLLMConvData(TestCase):

    def test_constructor_assertions(self):
        self.assertRaises(AssertionError, lambda: LLMConvData(
            context="You are in an online chatroom. You see the following post on a social media site:"
                    "'Homosexuality, transgender sex all are deviations from normal human behavior "
                    "as well.'",
            user_names=["Steve2001", "GeorgeBush78"],
            user_attributes=[
                ["African American"],
            ],
            user_instructions="Write a comment responding to the post. Insist on your opinion"
                              " unless convinced by other users."
                              " Try to follow the instructions of the moderator, if he intervenes."
                              " You are allowed to act frustrated or aggressively if provoked."))

    def test_from_json_file(self):
        data = LLMConvData.from_json_file("output/test1.json")
        assert len(data.user_names) != 0
        assert data.moderator_name is not None
        print(data)

        data2 = LLMConvData.from_json_file("output/test2.json")
        assert len(data2.user_names) != 0
        assert data2.moderator_name is None

        self.assertRaises(Exception, lambda: LLMConvData.from_json_file("output/test_invalid.json"))

    def test_to_json_file(self):
        data = LLMConvData(context="You are in an online chatroom. You see the following post on a social media site:"
                                   "'Homosexuality, transgender sex all are deviations from normal human behavior "
                                   "as well.'",
                           user_names=["Steve2001", "GeorgeBush78"],
                           user_attributes=[
                               ["African American"],
                               ["Typical", "average", "white", "American"]
                           ],
                           user_instructions="Write a comment responding to the post. Insist on your opinion"
                                             " unless convinced by other users."
                                             " Try to follow the instructions of the moderator, if he intervenes."
                                             " You are allowed to act frustrated or aggressively if provoked.",
                           moderator_name="moderator01",
                           moderator_attributes=["just", "strict"],
                           moderator_instructions="Intervene if one user dominates or veers off-topic. "
                                                  "Respond only if necessary. "
                                                  "Write '<No response>' if intervention is necessary."
                                                  " Be firm and threaten to discipline non-cooperating users.")
        data.to_json_file("output/test1.json")

        data2 = LLMConvData(context="You are in an online chatroom. You see the following post on a social media site:"
                                    "'Homosexuality, transgender sex all are deviations from normal human behavior "
                                    "as well.'",
                            user_names=["Steve2001", "GeorgeBush78"],
                            user_attributes=[
                                ["African American"],
                                ["Typical", "average", "white", "American"]
                            ],
                            user_instructions="Write a comment responding to the post. Insist on your opinion"
                                              " unless convinced by other users."
                                              " Try to follow the instructions of the moderator, if he intervenes."
                                              " You are allowed to act frustrated or aggressively if provoked.")
        data2.to_json_file("output/test2.json")
