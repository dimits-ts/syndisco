"""
Test suite for the Actor class.

Uses DummyModel — a deterministic stub that returns preset strings on
successive calls — so tests are hermetic and never hit a real LLM.
"""

import pytest
from .model import DummyModel
from syndisco import Actor


PERSONA_ALICE: dict[str, str] = {
    "name": "Alice",
    "background": "A software engineer who values clarity.",
    "tone": "concise",
}

CONTEXT = "Weekly team retrospective."
INSTRUCTIONS = "Share one thing that went well and one area to improve."


@pytest.fixture()
def dummy_model() -> DummyModel:
    return DummyModel(
        ["First response.", "Second response.", "Third response."]
    )


@pytest.fixture()
def actor(dummy_model: DummyModel):
    """A plain (non-annotator) Actor."""
    # import here so collection doesn't fail without the module

    return Actor(
        model=dummy_model,
        persona=PERSONA_ALICE,
        context=CONTEXT,
        instructions=INSTRUCTIONS,
        is_annotator=False,
        name="Alice",
    )


@pytest.fixture()
def annotator_actor(dummy_model: DummyModel):
    """An annotator Actor."""

    return Actor(
        model=dummy_model,
        persona=PERSONA_ALICE,
        context=CONTEXT,
        instructions=INSTRUCTIONS,
        is_annotator=True,
        name="Annotator",
    )


class TestActorConstruction:
    def test_name_is_stored(self, actor) -> None:
        assert actor.get_name() == "Alice"

    def test_default_name(self, dummy_model) -> None:

        unnamed = Actor(
            model=dummy_model,
            persona=PERSONA_ALICE,
            context=CONTEXT,
            instructions=INSTRUCTIONS,
        )
        assert unnamed.get_name() == "<Unnamed>"

    def test_is_annotator_false_by_default(self, dummy_model) -> None:

        a = Actor(
            model=dummy_model,
            persona=PERSONA_ALICE,
            context=CONTEXT,
            instructions=INSTRUCTIONS,
        )
        assert a.is_annotator is False  # adjust attr name if needed

    def test_is_annotator_true_when_set(self, annotator_actor) -> None:
        assert annotator_actor.is_annotator is True


class TestGetSystemPrompt:
    def test_returns_string(self, actor) -> None:
        result = actor.get_system_prompt()
        assert isinstance(result, str)

    def test_contains_context(self, actor) -> None:
        assert CONTEXT in actor.get_system_prompt()

    def test_contains_instructions(self, actor) -> None:
        assert INSTRUCTIONS in actor.get_system_prompt()

    def test_contains_persona_name(self, actor) -> None:
        assert PERSONA_ALICE["name"] in actor.get_system_prompt()

    def test_annotator_prompt_differs_from_participant_prompt(
        self, actor, annotator_actor
    ) -> None:
        """
        Annotators and participants should receive distinct system prompts.
        """
        assert actor.get_system_prompt() != annotator_actor.get_system_prompt()

    def test_prompt_is_non_empty(self, actor) -> None:
        assert actor.get_system_prompt().strip() != ""


class TestGetUserPrompt:
    def test_returns_string_with_no_history(self, actor) -> None:
        result = actor.get_user_prompt(history=None)
        assert isinstance(result, str)

    def test_returns_string_with_empty_history(self, actor) -> None:
        result = actor.get_user_prompt(history=[])
        assert isinstance(result, str)

    def test_returns_string_with_history(self, actor) -> None:
        history = ["Alice: I agree.", "Bob: Me too."]
        result = actor.get_user_prompt(history=history)
        assert isinstance(result, str)

    def test_history_messages_appear_in_prompt(self, actor) -> None:
        history = ["Alice: Great sprint.", "Bob: Agreed."]
        prompt = actor.get_user_prompt(history=history)
        assert "Alice: Great sprint." in prompt
        assert "Bob: Agreed." in prompt

    def test_no_history_none_differs_from_with_history(self, actor) -> None:
        no_history = actor.get_user_prompt(history=None)
        with_history = actor.get_user_prompt(history=["Bob: Hello."])
        assert no_history != with_history

    def test_annotator_user_prompt_differs_from_participant(
        self, actor, annotator_actor
    ) -> None:
        history = ["Alice: Something happened."]
        assert actor.get_user_prompt(
            history
        ) != annotator_actor.get_user_prompt(history)


class TestSpeak:
    def test_returns_string_no_history(self, actor) -> None:
        result = actor.speak(history=None)
        assert isinstance(result, str)

    def test_returns_string_with_history(self, actor) -> None:
        result = actor.speak(history=["Bob: Hi."])
        assert isinstance(result, str)

    def test_uses_model_output(self, actor, dummy_model) -> None:
        """speak() should return exactly what the model produces."""
        assert actor.speak() == "First response."

    def test_successive_calls_return_successive_model_outputs(
        self, actor, dummy_model
    ) -> None:
        first = actor.speak()
        second = actor.speak()
        assert first != second

    def test_model_called_exactly_once_per_speak(
        self, actor, dummy_model
    ) -> None:
        actor.speak()
        assert dummy_model.call_count == 1
        actor.speak()
        assert dummy_model.call_count == 2

    def test_speak_passes_system_prompt_to_model(self, dummy_model) -> None:
        """
        The system prompt returned by get_system_prompt() reaches the model.
        """

        a = Actor(
            model=dummy_model,
            persona=PERSONA_ALICE,
            context=CONTEXT,
            instructions=INSTRUCTIONS,
            name="Alice",
        )

        captured: dict = {}
        original_prompt = dummy_model.prompt

        def capturing_prompt(system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return original_prompt(system, user)

        dummy_model.prompt = capturing_prompt
        a.speak(history=None)

        assert captured.get("system") == a.get_system_prompt()

    def test_speak_passes_user_prompt_to_model(self, dummy_model) -> None:
        """The user prompt returned by get_user_prompt() reaches the model."""

        history = ["Carol: First message."]
        a = Actor(
            model=dummy_model,
            persona=PERSONA_ALICE,
            context=CONTEXT,
            instructions=INSTRUCTIONS,
            name="Alice",
        )

        captured: dict = {}
        original_prompt = dummy_model.prompt

        def capturing_prompt(system: str, user: str) -> str:
            captured["system"] = system
            captured["user"] = user
            return original_prompt(system, user)

        dummy_model.prompt = capturing_prompt
        a.speak(history=history)

        assert captured.get("user") == a.get_user_prompt(history=history)

    def test_speak_with_single_item_history(self, actor) -> None:
        result = actor.speak(history=["Bob: One message."])
        assert isinstance(result, str) and len(result) > 0

    def test_speak_with_long_history(self, actor) -> None:
        history = [f"Speaker{i}: Message {i}." for i in range(50)]
        result = actor.speak(history=history)
        assert isinstance(result, str)

    def test_annotator_speak_produces_output(self, annotator_actor) -> None:
        result = annotator_actor.speak(history=["Alice: Interesting point."])
        assert isinstance(result, str) and len(result) > 0


class TestGetName:
    def test_returns_string(self, actor) -> None:
        assert isinstance(actor.get_name(), str)

    def test_correct_name(self, actor) -> None:
        assert actor.get_name() == "Alice"

    def test_name_unchanged_after_speak(self, actor) -> None:
        actor.speak()
        assert actor.get_name() == "Alice"
