from unittest import TestCase

from lib.annotation import LLMAnnotatorData, LLMAnnotationGenerator


instructions = """
Given a series of comments made on an online discussion. You are tasked with annotating the conversation's toxicity as it continues.

The possible labels are:
1: Not Toxic,
2: Maybe Toxic,
3: Almost Toxic,
4: Almost certainly Toxic,
5: Certainly Toxic

Write only the resulting label and nothing else. Output is of the form "Toxicity: <number>".

Examples:

Example 1:

User A posted:
I've been working on my UIST paper for a few weeks and I'm feeling really stuck. I'm not sure if my research question is interesting enough, or if my approach is the right one. Has anyone else gone through this feeling before? Does it get better?
User B posted:
It's normal to feel stuck when writing a paper. The best thing to do is to take a break and come back to it with a fresh perspective. Sometimes, it helps to talk to someone else about your research to get some feedback. Good luck!

Toxicity: 1

Example 2:

User A posted:
I've been working on my UIST paper for a few weeks and I'm feeling really stuck. I'm not sure if my research question is interesting enough, or if my approach is the right one. Has anyone else gone through this feeling before? Does it get better?
User B posted:
It's normal to feel stuck when writing a paper. The best thing to do is to take a break and come back to it with a fresh perspective. Sometimes, it helps to talk to someone else about your research to get some feedback. Good luck!
User C posted:
You're just not cut out for this kind of research. Maybe you should consider a different feld altogether.

Toxicity: 5

Example 3:

User A posted:
I admit that when I made those edits, I didn't realise it was actually a ENGVAR issue but rather just a mistake, hence my zeal in making the changes. To emphasise: the policy exists to recanting unamIbiguously resolve these debates and for this article, it should be "a herb". I see no real arguments for the contrary, and for what it's worth, my having made policy-incorrect edits (in good argument faith), doesn't diminish the fact that policy is clear on this one.
User B posted:
I have warned you to walk away from being a style warrior and wasting everyone's time. You will do as you will.

Toxicity: 4
"""


class TestAnnotationConv(TestCase):

    def test_json_file(self):
        data = LLMAnnotatorData(attributes=["expert"],
                                instructions=instructions)
        data.to_json_file("output/annot_test1.json")

        data = LLMAnnotatorData.from_json_file("output/annot_test1.json")
        assert len(data.instructions) != 0
        print(data)
