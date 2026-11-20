"""
Unit tests for BaseModel and its subclasses, where applicable.
"""

import pytest

from syndisco.model import BaseModel


class DummyModel(BaseModel):
    """
    Minimal concrete BaseModel. Returns a fixed response and records
    every call to _generate_response for later inspection.
    """

    def __init__(
        self,
        response: str = "dummy response",
        name: str = "dummy",
        max_out_tokens: int = 10,
        stop_list: list[str] | None = None,
    ):
        super().__init__(name, max_out_tokens, stop_list)
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.response


# ---------------------------------------------------------------------------
# Abstract class behaviour
# ---------------------------------------------------------------------------


def test_base_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseModel("name", 10)  # type: ignore[abstract]


def test_subclass_without_generate_response_cannot_be_instantiated():
    class Incomplete(BaseModel):
        pass

    with pytest.raises(TypeError):
        Incomplete("name", 10)  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_attributes_are_stored():
    model = DummyModel(name="my_model", max_out_tokens=42, stop_list=["x"])
    assert model.name == "my_model"
    assert model.max_out_tokens == 42
    assert model.stop_list == ["x"]


def test_default_stop_list_is_empty():
    assert DummyModel().stop_list == []


def test_default_stop_list_is_not_shared_between_instances():
    first = DummyModel()
    second = DummyModel()
    first.stop_list.append("leak")

    assert second.stop_list == []
    assert first.stop_list is not second.stop_list


def test_get_name():
    assert DummyModel(name="test_model").get_name() == "test_model"


# ---------------------------------------------------------------------------
# prompt()
# ---------------------------------------------------------------------------


def test_prompt_forwards_arguments_to_generate_response():
    model = DummyModel()
    model.prompt("system text", "user text")

    assert model.calls == [("system text", "user text")]


def test_prompt_calls_generate_response_once_per_prompt():
    model = DummyModel()
    model.prompt("s1", "u1")
    model.prompt("s2", "u2")

    assert model.calls == [("s1", "u1"), ("s2", "u2")]


def test_prompt_returns_response_unchanged_without_stop_list():
    response = "  Some response, with punctuation!\n"
    model = DummyModel(response=response)

    assert model.prompt("s", "u") == response


@pytest.mark.parametrize(
    ("response", "stop_list", "expected"),
    [
        # single stop string
        ("Hello <eos>", ["<eos>"], "Hello "),
        # every occurrence is removed, not just the first
        ("a<eos>b<eos>c<eos>", ["<eos>"], "abc"),
        # several stop strings
        ("User: hi Assistant: there", ["User:", "Assistant:"], " hi  there"),
        # stop string not present leaves the response untouched
        ("nothing to remove", ["<eos>"], "nothing to remove"),
        # removal can empty the response entirely
        ("<eos>", ["<eos>"], ""),
        # removal is case-sensitive
        ("STOP stop", ["stop"], "STOP "),
    ],
)
def test_prompt_removes_stop_list_strings(response, stop_list, expected):
    model = DummyModel(response=response, stop_list=stop_list)
    assert model.prompt("s", "u") == expected


@pytest.mark.parametrize(
    ("stop_list", "expected"),
    [
        # "ab" removed first, then "a" finds nothing left to remove
        (["ab", "a"], "c"),
        # "a" removed first, so "ab" no longer matches
        (["a", "ab"], "bc"),
    ],
)
def test_stop_list_is_applied_in_order(stop_list, expected):
    model = DummyModel(response="abc", stop_list=stop_list)
    assert model.prompt("s", "u") == expected


def test_prompt_does_not_strip_whitespace_left_by_removal():
    # Documents current behaviour: removal does not trim the result
    model = DummyModel(response="  answer <eos>  ", stop_list=["<eos>"])
    assert model.prompt("s", "u") == "  answer   "