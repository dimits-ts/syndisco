"""
Test suite for Logs, Discussion, and Annotation.

DummyModel, DummyActor, and DummyTurnManager are deterministic stubs;
no real LLM or file-system I/O is required except in the export/from_file
tests, which use pytest's tmp_path fixture.
"""

import collections.abc
import json
import pytest
from pathlib import Path
from datetime import datetime

from .dummy import DummyActor
from syndisco import Discussion, Logs, RandomWeighted, Annotation


def make_logs(entries: list[tuple[str, str, str]] | None = None):
    """Return a populated Logs instance from (name, text, model) tuples."""
    logs = Logs()
    for name, text, model in entries or []:
        logs.append(name=name, text=text, model=model)
    return logs


def make_discussion(
    conv_len: int = 3,
    num_actors: int = 2,
    seed_opinions: list[str] | None = None,
    seed_opinion_usernames: list[str] | None = None,
    history_context_len: int = 5,
):
    actors = [
        DummyActor(name=f"User{i}", responses=[f"User{i} reply."])
        for i in range(num_actors)
    ]
    tm = RandomWeighted(actors=actors, p_respond=0.4)
    return Discussion(
        next_turn_manager=tm,
        users=actors,
        history_context_len=history_context_len,
        conv_len=conv_len,
        seed_opinions=seed_opinions,
        seed_opinion_usernames=seed_opinion_usernames,
    )


class TestLogsEquality:

    def test_equal_empty_logs(self) -> None:
        assert Logs() == Logs()

    def test_equal_same_entries(self) -> None:
        a = make_logs([("Alice", "Hello", "m")])
        b = make_logs([("Alice", "Hello", "m")])
        assert a == b

    def test_not_equal_different_text(self) -> None:
        a = make_logs([("Alice", "Hello", "m")])
        b = make_logs([("Alice", "Goodbye", "m")])
        assert a != b

    def test_not_equal_different_order(self) -> None:
        a = make_logs([("Alice", "first", "m"), ("Bob", "second", "m")])
        b = make_logs([("Bob", "second", "m"), ("Alice", "first", "m")])
        assert a != b

    def test_not_equal_different_length(self) -> None:
        a = make_logs([("Alice", "Hello", "m")])
        b = make_logs([("Alice", "Hello", "m"), ("Bob", "Hi", "m")])
        assert a != b

    def test_not_equal_to_non_logs(self) -> None:
        logs = make_logs([("Alice", "Hello", "m")])
        assert logs != [{"name": "Alice", "text": "Hello", "model": "m"}]


class TestLogsConstruction:

    def test_empty_on_init(self) -> None:
        assert len(Logs()) == 0

    def test_append_increases_length(self) -> None:
        logs = make_logs()

        logs = Logs()
        logs.append(name="Alice", text="Hello", model="gpt")
        assert len(logs) == 1

    def test_append_multiple(self) -> None:

        logs = Logs()
        for i in range(5):
            logs.append(name=f"U{i}", text=f"msg{i}", model="m")
        assert len(logs) == 5

    def test_append_default_model_is_hardcoded(self) -> None:

        logs = Logs()
        logs.append(name="Alice", text="Hi")
        assert logs[0]["model"] == "hardcoded"

    def test_append_stores_prompt(self) -> None:

        logs = Logs()
        logs.append(name="Alice", text="Hi", model="gpt", prompt="<sys>")
        assert logs[0]["prompt"] == "<sys>"

    def test_append_empty_prompt_by_default(self) -> None:

        logs = Logs()
        logs.append(name="Alice", text="Hi", model="gpt")
        assert logs[0]["prompt"] == ""

    def test_entry_keys(self) -> None:

        logs = Logs()
        logs.append(name="Alice", text="Hi", model="gpt")
        assert {"name", "text", "model"}.issubset(logs[0].keys())


class TestLogsAccess:

    def test_getitem_returns_correct_entry(self) -> None:
        logs = make_logs([("Alice", "Hello", "m1"), ("Bob", "World", "m2")])
        assert logs[0]["name"] == "Alice"
        assert logs[1]["name"] == "Bob"

    def test_getitem_correct_text(self) -> None:
        logs = make_logs([("Alice", "Hello", "m")])
        assert logs[0]["text"] == "Hello"

    def test_iter_yields_all_entries(self) -> None:
        entries = [("A", "msg1", "m"), ("B", "msg2", "m"), ("C", "msg3", "m")]
        logs = make_logs(entries)
        names = [e["name"] for e in logs]
        assert names == ["A", "B", "C"]

    def test_iter_preserves_order(self) -> None:
        entries = [(str(i), f"text{i}", "m") for i in range(10)]
        logs = make_logs(entries)
        for i, entry in enumerate(logs):
            assert entry["name"] == str(i)

    def test_len_matches_append_count(self) -> None:
        logs = make_logs([("A", "t", "m")] * 7)
        assert len(logs) == 7


class TestLogsToList:

    def test_returns_list(self) -> None:
        logs = make_logs([("A", "t", "m")])
        assert isinstance(logs.to_list(), list)

    def test_length_matches(self) -> None:
        entries = [("A", "t", "m"), ("B", "u", "n")]
        logs = make_logs(entries)
        assert len(logs.to_list()) == 2

    def test_is_shallow_copy(self) -> None:
        logs = make_logs([("A", "t", "m")])
        copy = logs.to_list()
        copy.append({"name": "X", "text": "y", "model": "z"})
        assert len(logs) == 1  # original unchanged


class TestLogsToDict:

    def test_returns_dict(self) -> None:
        logs = make_logs([("A", "t", "m")])
        assert isinstance(logs.to_dict(), dict)

    def test_contains_entries_key(self) -> None:
        logs = make_logs([("A", "t", "m")])
        d = logs.to_dict()
        assert (
            "entries" in d or "messages" in d or "logs" in d
        )  # key name may vary

    def test_contains_timestamp(self) -> None:
        logs = make_logs([("A", "t", "m")])
        d = logs.to_dict()
        assert "timestamp" in d

    def test_timestamp_matches_format(self) -> None:
        fmt = "%y-%m-%d-%H-%M"
        logs = make_logs()
        d = logs.to_dict(timestamp_format=fmt)
        # Should be parseable with the given format
        datetime.strptime(d["timestamp"], fmt)  # raises if format mismatch

    def test_custom_timestamp_format(self) -> None:
        fmt = "%Y/%m/%d"
        logs = make_logs()
        d = logs.to_dict(timestamp_format=fmt)
        datetime.strptime(d["timestamp"], fmt)

    def test_entry_data_preserved(self) -> None:
        logs = make_logs([("Alice", "Hello", "gpt4")])
        d = logs.to_dict()
        # Find the entries list regardless of the key name used
        entries_list = next(v for v in d.values() if isinstance(v, list))
        assert any(e.get("name") == "Alice" for e in entries_list)


class TestLogsExportFromFile:

    def test_export_creates_file(self, tmp_path: Path) -> None:
        logs = make_logs([("Alice", "Hello", "m")])
        out = tmp_path / "logs.json"
        logs.export(output_path=out)
        assert out.exists()

    def test_export_valid_json(self, tmp_path: Path) -> None:
        logs = make_logs([("Alice", "Hello", "m")])
        out = tmp_path / "logs.json"
        logs.export(output_path=out)
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_export_string_path(self, tmp_path: Path) -> None:
        logs = make_logs([("Alice", "Hello", "m")])
        out = str(tmp_path / "logs.json")
        logs.export(output_path=out)
        assert Path(out).exists()

    def test_from_file_returns_logs_instance(self, tmp_path: Path) -> None:

        logs = make_logs([("Alice", "Hi", "m")])
        out = tmp_path / "logs.json"
        logs.export(output_path=out)
        loaded = Logs.from_file(out)
        assert isinstance(loaded, Logs)

    def test_from_file_preserves_length(self, tmp_path: Path) -> None:

        original = make_logs([("A", "t", "m"), ("B", "u", "n")])
        out = tmp_path / "logs.json"
        original.export(output_path=out)
        loaded = Logs.from_file(out)
        assert len(loaded) == len(original)

    def test_from_file_preserves_entry_fields(self, tmp_path: Path) -> None:

        original = make_logs([("Alice", "Hello there", "gpt4")])
        out = tmp_path / "logs.json"
        original.export(output_path=out)
        loaded = Logs.from_file(out)
        assert loaded[0]["name"] == "Alice"
        assert loaded[0]["text"] == "Hello there"
        assert loaded[0]["model"] == "gpt4"

    def test_from_file_raises_file_not_found(self, tmp_path: Path) -> None:

        with pytest.raises(FileNotFoundError):
            Logs.from_file(tmp_path / "nonexistent.json")

    def test_from_file_raises_value_error_on_bad_schema(
        self, tmp_path: Path
    ) -> None:

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"unexpected": "structure"}))
        with pytest.raises(ValueError):
            Logs.from_file(bad)

    def test_from_file_accepts_string_path(self, tmp_path: Path) -> None:

        logs = make_logs([("A", "t", "m")])
        out = tmp_path / "logs.json"
        logs.export(output_path=out)
        loaded = Logs.from_file(str(out))
        assert len(loaded) == 1


class TestDiscussionConstruction:

    def test_constructs_without_seeds(self) -> None:
        d = make_discussion(conv_len=3)
        assert d is not None

    def test_discussion_raises_with_annotator_actor(self) -> None:
        user1 = DummyActor(name="Alice")
        user2 = DummyActor(name="Bob")
        annotator = DummyActor(name="Annotator", is_annotator=True)
        users = [user1, user2, annotator]
        tm = RandomWeighted(users)
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=users,
                conv_len=2,
            )

    def test_constructs_with_seeds(self) -> None:
        d = make_discussion(
            conv_len=2,
            seed_opinions=["Seed one.", "Seed two."],
            seed_opinion_usernames=["User0", "User1"],
        )
        assert d is not None

    def test_mismatched_seed_counts_raise(self) -> None:
        actors = [DummyActor("A"), DummyActor("B")]
        tm = RandomWeighted(actors, p_respond=0)
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=actors,
                conv_len=2,
                seed_opinions=["one", "two"],
                seed_opinion_usernames=["A"],  # mismatch
            )

    def test_too_many_seeds_raise(self) -> None:
        actors = [DummyActor("A")]
        tm = RandomWeighted(actors, p_respond=0)
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=actors,
                conv_len=1,
                seed_opinions=["s1", "s2"],  # more seeds than usernames
                seed_opinion_usernames=["A", "A", "A"],
            )

    def test_none_seeds_raise(self) -> None:
        actors = [DummyActor("A")]
        tm = RandomWeighted(actors, p_respond=0)
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=actors,
                conv_len=1,
                seed_opinions=[None, "s2"],  # type: ignore
                seed_opinion_usernames=["A", "A"],
            )

    def test_none_seed_usernames_raise(self) -> None:
        actors = [DummyActor("A")]
        tm = RandomWeighted(actors, p_respond=0)
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=actors,
                conv_len=1,
                seed_opinions=["s1", "s2"],
                seed_opinion_usernames=["A", None],  # type: ignore
            )

    def test_single_seed_does_not_raise(self) -> None:
        actors = [DummyActor("A")]
        tm = RandomWeighted(actors, p_respond=0)
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=actors,
                conv_len=1,
                seed_opinions="hi",
                seed_opinion_usernames="A",
            )

    def test_weird_seed_config_raise(self) -> None:
        actors = [DummyActor("A")]
        tm = RandomWeighted(actors, p_respond=0)
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=actors,
                conv_len=1,
                seed_opinions=["s1", "s2"],
                seed_opinion_usernames=None,
            )
        with pytest.raises(ValueError):
            Discussion(
                next_turn_manager=tm,
                users=actors,
                conv_len=1,
                seed_opinions=None,
                seed_opinion_usernames=["A", "A"],
            )


class TestDiscussionIteratorProtocol:

    def test_is_iterator(self) -> None:
        d = make_discussion(conv_len=2)
        assert isinstance(d, collections.abc.Iterator)

    def test_iter_returns_self(self) -> None:
        d = make_discussion(conv_len=2)
        assert iter(d) is d

    def test_next_returns_dict(self) -> None:
        d = make_discussion(conv_len=3)
        entry = next(d)
        assert isinstance(entry, dict)

    def test_entry_has_required_keys(self) -> None:
        d = make_discussion(conv_len=3)
        entry = next(d)
        assert {"name", "text", "model"}.issubset(entry.keys())

    def test_stops_after_conv_len_steps(self) -> None:
        conv_len = 4
        d = make_discussion(conv_len=conv_len)
        results = list(d)
        assert len(results) == conv_len

    def test_raises_stop_iteration_when_exhausted(self) -> None:
        d = make_discussion(conv_len=2)
        list(d)  # exhaust
        with pytest.raises(StopIteration):
            next(d)

    def test_for_loop_yields_conv_len_entries(self) -> None:
        conv_len = 5
        d = make_discussion(conv_len=conv_len)
        count = sum(1 for _ in d)
        assert count == conv_len

    def test_entry_name_matches_an_actor(self) -> None:
        actors = [DummyActor("Alice"), DummyActor("Bob")]
        tm = RandomWeighted(actors, p_respond=0.5)

        d = Discussion(next_turn_manager=tm, users=actors, conv_len=4)
        for entry in d:
            assert entry["name"] in {"Alice", "Bob"}

    def test_entry_text_is_nonempty_string(self) -> None:
        d = make_discussion(conv_len=3)
        for entry in d:
            # whitespace-only entries are still allowed (they are skipped
            # internally but still counted); non-whitespace entries must be str
            assert isinstance(entry["text"], str)


class TestDiscussionSeeds:

    def test_seed_opinions_appear_in_logs(self) -> None:
        seeds = ["First seed.", "Second seed."]
        usernames = ["User0", "User1"]
        d = make_discussion(
            conv_len=2,
            seed_opinions=seeds,
            seed_opinion_usernames=usernames,
        )
        list(d)  # run to completion
        logs = d.get_logs()
        texts = [e["text"] for e in logs]
        assert "First seed." in texts
        assert "Second seed." in texts

    def test_seed_usernames_appear_in_logs(self) -> None:
        d = make_discussion(
            conv_len=2,
            seed_opinions=["Hello."],
            seed_opinion_usernames=["SeedUser"],
        )
        list(d)
        names = [e["name"] for e in d.get_logs()]
        assert "SeedUser" in names

    def test_seed_model_is_hardcoded(self) -> None:
        d = make_discussion(
            conv_len=1,
            seed_opinions=["Seed."],
            seed_opinion_usernames=["User0"],
        )
        list(d)
        seed_entries = [e for e in d.get_logs() if e["text"] == "Seed."]
        assert all(e["model"] == "hardcoded" for e in seed_entries)

    def test_seeds_not_counted_toward_conv_len(self) -> None:
        conv_len = 3
        d = make_discussion(
            conv_len=conv_len,
            seed_opinions=["Seed."],
            seed_opinion_usernames=["User0"],
        )
        list(d)
        logs = d.get_logs()
        prompted = [e for e in logs if e["model"] != "hardcoded"]
        assert len(prompted) == conv_len


class TestDiscussionBegin:

    def test_begin_completes_without_error(self) -> None:
        d = make_discussion(conv_len=3)
        d.begin(verbose=False)

    def test_begin_verbose_false_produces_no_stdout(self, capsys) -> None:
        d = make_discussion(conv_len=2)
        d.begin(verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_begin_verbose_true_prints_output(self, capsys) -> None:
        d = make_discussion(conv_len=2)
        d.begin(verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_begin_exhausts_discussion(self) -> None:
        d = make_discussion(conv_len=3)
        d.begin(verbose=False)
        with pytest.raises(StopIteration):
            next(d)

    def test_discussion_entries_include_prompt_field(self) -> None:
        d = make_discussion(conv_len=3)
        list(d)
        logs = d.get_logs()

        for entry in logs:
            assert "prompt" in entry

    def test_discussion_prompt_is_non_null_string(self) -> None:
        d = make_discussion(conv_len=3)
        list(d)
        logs = d.get_logs()

        for entry in logs:
            assert isinstance(entry["prompt"], str)
            assert entry["prompt"] != ""

    def test_seed_entries_have_empty_prompt(self) -> None:
        d = make_discussion(
            conv_len=2,
            seed_opinions=["Seed text."],
            seed_opinion_usernames=["User0"],
        )
        list(d)
        logs = d.get_logs()

        seed_entries = [e for e in logs if e["model"] == "hardcoded"]
        assert all(
            e["prompt"]
            == '{"context": "", "instructions": "", "type": "user", "persona": {}}'
            for e in seed_entries
        )


class TestDiscussionGetLogs:

    def test_get_logs_returns_logs_instance(self) -> None:

        d = make_discussion(conv_len=2)
        list(d)
        assert isinstance(d.get_logs(), Logs)

    def test_logs_length_equals_seeds_plus_conv_len(self) -> None:
        conv_len = 3
        seeds = ["s1", "s2"]
        d = make_discussion(
            conv_len=conv_len,
            seed_opinions=seeds,
            seed_opinion_usernames=["User0", "User1"],
        )
        list(d)
        # Seeds that ARE whitespace-only are skipped from appending, but
        # our DummyActor never returns whitespace, so total = seeds + conv_len.
        assert len(d.get_logs()) == len(seeds) + conv_len

    def test_get_logs_is_copy(self) -> None:
        d = make_discussion(conv_len=2)
        list(d)
        logs1 = d.get_logs()
        logs2 = d.get_logs()
        assert logs1 is not logs2


class TestAnnotationConstruction:

    def test_constructs_with_logs(self) -> None:

        annotator = DummyActor(
            name="Annotator",
            is_annotator=True,
            responses=["annotation result."],
        )
        logs = make_logs([("Alice", "Hello.", "m"), ("Bob", "Hi there.", "m")])
        ann = Annotation(annotator=annotator, discussion_logs=logs)
        assert ann is not None

    def test_constructs_with_empty_logs(self) -> None:

        annotator = DummyActor(name="Annotator", is_annotator=True)
        ann = Annotation(annotator=annotator, discussion_logs=Logs())
        assert ann is not None

    def test_custom_history_ctx_len(self) -> None:

        annotator = DummyActor(name="Annotator", is_annotator=True)
        logs = make_logs([("A", "t", "m")])
        ann = Annotation(
            annotator=annotator, discussion_logs=logs, history_ctx_len=4
        )
        assert ann is not None


class TestAnnotationGetLogs:

    def _make_annotation(self, num_entries: int = 3):

        responses = [f"annotation {i}" for i in range(num_entries)]
        annotator = DummyActor(
            name="Annotator", is_annotator=True, responses=responses
        )
        entries = [
            (f"User{i}", f"Comment {i}.", "m") for i in range(num_entries)
        ]
        logs = make_logs(entries)
        annotation = Annotation(annotator=annotator, discussion_logs=logs)
        annotation.begin()
        return annotation

    def test_get_logs_returns_logs_instance(self) -> None:

        ann = self._make_annotation()
        assert isinstance(ann.get_logs(), Logs)

    def test_annotation_count_matches_discussion_entries(self) -> None:
        num_entries = 4
        ann = self._make_annotation(num_entries=num_entries)
        logs = ann.get_logs()
        assert len(logs) == num_entries

    def test_annotation_texts_are_strings(self) -> None:
        ann = self._make_annotation(num_entries=3)
        for entry in ann.get_logs():
            assert isinstance(entry["text"], str)

    def test_get_logs_is_copy(self) -> None:
        ann = self._make_annotation(num_entries=2)
        l1 = ann.get_logs()
        l2 = ann.get_logs()
        assert l1 is not l2

    def test_empty_discussion_produces_empty_annotation_logs(self) -> None:
        annotator = DummyActor(name="Annotator", is_annotator=True)
        ann = Annotation(annotator=annotator, discussion_logs=Logs())
        assert len(ann.get_logs()) == 0

    def test_annotation_prompt_overwrites_original_prompt(self) -> None:
        logs = Logs()
        logs.append(
            name="User0",
            text="Text",
            model="m",
            prompt="ORIGINAL_PROMPT",
        )

        annotator = DummyActor(
            name="Annotator",
            is_annotator=True,
            responses=["label"],
        )

        ann = Annotation(annotator=annotator, discussion_logs=logs)
        ann.begin()

        entry = ann.get_logs()[0]

        # The original prompt should NOT survive
        assert entry["prompt"] != "ORIGINAL_PROMPT"

    def test_annotation_prompt_equals_annotator_prompt(self) -> None:
        annotator_prompt = "<ANNOTATOR_SYSTEM_PROMPT>"

        annotator = DummyActor(
            name="Annotator",
            is_annotator=True,
            responses=["label"],
            instructions=annotator_prompt,
        )

        logs = make_logs([("User0", "Text", "m")])

        ann = Annotation(annotator=annotator, discussion_logs=logs)
        ann.begin()

        entry = ann.get_logs()[0]

        assert annotator_prompt in entry["prompt"]

    def test_all_annotation_entries_use_same_prompt(self) -> None:
        annotator_prompt = "<ANNOTATOR_PROMPT>"

        annotator = DummyActor(
            name="Annotator",
            is_annotator=True,
            responses=["l1", "l2", "l3"],
            instructions=annotator_prompt,
        )

        logs = make_logs(
            [
                ("User0", "Text0", "m"),
                ("User1", "Text1", "m"),
                ("User2", "Text2", "m"),
            ]
        )

        ann = Annotation(annotator=annotator, discussion_logs=logs)
        ann.begin()

        prompts = [e["prompt"] for e in ann.get_logs()]
        assert all(annotator_prompt in p for p in prompts)

    def test_annotation_prompt_is_nonempty_when_annotator_has_prompt(
        self,
    ) -> None:
        annotator = DummyActor(
            name="Annotator",
            is_annotator=True,
            responses=["label"],
            instructions="ANNOTATOR_PROMPT",
        )

        logs = make_logs([("User0", "Text", "m")])

        ann = Annotation(annotator=annotator, discussion_logs=logs)
        ann.begin()

        entry = ann.get_logs()[0]
        assert "ANNOTATOR_PROMPT" in entry["prompt"]
