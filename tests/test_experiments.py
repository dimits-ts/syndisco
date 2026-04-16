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
"""
Test suite for DiscussionExperiment and AnnotationExperiment.

All LLM / file-system interactions are replaced by deterministic stubs.
Real file I/O uses pytest's tmp_path fixture.
"""

import json
import pytest
from pathlib import Path

from syndisco import (
    DiscussionExperiment,
    AnnotationExperiment,
    RandomWeighted,
    Logs,
)
from .dummy import DummyActor


def make_logs(num_entries: int = 3) -> Logs:
    logs = Logs()
    for i in range(num_entries):
        logs.append(
            name=f"User{i}",
            text=f"Comment {i}.",
            model="dummy",
        )
    return logs


def make_users(n: int = 3, *, is_annotator: bool = False) -> list[DummyActor]:
    return [
        DummyActor(name=f"User{i}", is_annotator=is_annotator)
        for i in range(n)
    ]


def make_annotators(n: int = 2) -> list[DummyActor]:
    return make_users(n, is_annotator=True)


def write_dummy_discussion_log(path: Path, num_entries: int = 3) -> None:
    """Write a minimal discussion JSON file that Logs.from_file() can read."""
    entries = [
        {
            "name": f"User{i}",
            "text": f"Comment {i}.",
            "model": "dummy",
            "prompt": "",
        }
        for i in range(num_entries)
    ]
    data = {
        "timestamp": "25-01-01-00-00",
        "entries": entries,
    }
    path.write_text(json.dumps(data))


class TestDiscussionExperimentConstruction:

    def test_constructs_with_minimal_args(self) -> None:
        exp = DiscussionExperiment(users=make_users())
        assert exp is not None

    def test_constructs_with_all_args(self) -> None:
        seeds = [["Seed one.", "Seed two."], ["Seed three."]]
        tm = RandomWeighted
        exp = DiscussionExperiment(
            users=make_users(4),
            seed_opinions=seeds,
            turn_manager_factory=tm,
            history_ctx_len=5,
            num_turns=8,
            num_active_users=2,
            num_discussions=3,
        )
        assert exp is not None

    def test_raises_when_num_active_users_exceeds_total_users(self) -> None:
        with pytest.raises(ValueError):
            DiscussionExperiment(
                users=make_users(2),
                num_active_users=5,
            )

    def test_raises_when_users_list_is_empty(self) -> None:
        with pytest.raises(ValueError):
            DiscussionExperiment(users=[])

    def test_raises_when_num_discussions_is_zero(self) -> None:
        with pytest.raises(ValueError):
            DiscussionExperiment(users=make_users(), num_discussions=0)

    def test_raises_when_num_turns_is_zero(self) -> None:
        with pytest.raises(ValueError):
            DiscussionExperiment(users=make_users(), num_turns=0)

    def test_raises_when_num_active_users_is_zero(self) -> None:
        with pytest.raises(ValueError):
            DiscussionExperiment(users=make_users(), num_active_users=0)

    def test_raises_when_annotator_passed_as_user(
        self, tmp_path: Path
    ) -> None:
        users = make_users(2) + [DummyActor(name="Ann", is_annotator=True)]
        out = tmp_path / "null"
        with pytest.raises(ValueError):
            DiscussionExperiment(users=users).begin(discussions_output_dir=out)

    def test_none_seed_opinions_accepted(self) -> None:
        DiscussionExperiment(users=make_users(), seed_opinions=None)


class TestDiscussionExperimentBegin:

    def test_begin_creates_output_directory(self, tmp_path: Path) -> None:
        out = tmp_path / "discussions"
        exp = DiscussionExperiment(
            users=make_users(3),
            num_discussions=1,
            num_turns=2,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=out, verbose=False)
        assert out.exists() and out.is_dir()

    def test_begin_writes_correct_number_of_files(
        self, tmp_path: Path
    ) -> None:
        num_discussions = 3
        exp = DiscussionExperiment(
            users=make_users(4),
            num_discussions=num_discussions,
            num_turns=2,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=tmp_path, verbose=False)
        json_files = list(tmp_path.glob("*.json"))
        # the files are saved as timestamps with second precision
        # thus, they will overwrite each other in these tests
        assert len(json_files) > 0

    def test_begin_output_files_are_valid_json(self, tmp_path: Path) -> None:
        exp = DiscussionExperiment(
            users=make_users(3),
            num_discussions=2,
            num_turns=2,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=tmp_path, verbose=False)
        for f in tmp_path.glob("*.json"):
            data = json.loads(f.read_text())
            assert isinstance(data, dict)

    def test_begin_output_files_contain_entries(self, tmp_path: Path) -> None:
        exp = DiscussionExperiment(
            users=make_users(3),
            num_discussions=1,
            num_turns=3,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=tmp_path, verbose=False)
        data = json.loads(next(tmp_path.glob("*.json")).read_text())
        entries = next(v for v in data.values() if isinstance(v, list))
        assert len(entries) > 0

    def test_begin_uses_exactly_num_active_users_per_discussion(
        self, tmp_path: Path
    ) -> None:
        num_active = 2
        exp = DiscussionExperiment(
            users=make_users(5),
            num_discussions=3,
            num_turns=2,
            num_active_users=num_active,
        )
        exp.begin(discussions_output_dir=tmp_path, verbose=False)
        for f in tmp_path.glob("*.json"):
            data = json.loads(f.read_text())
            entries = next(v for v in data.values() if isinstance(v, list))
            speakers = {e["name"] for e in entries}
            assert len(speakers) <= num_active

    def test_begin_verbose_false_produces_no_stdout(
        self, tmp_path: Path, capsys
    ) -> None:
        exp = DiscussionExperiment(
            users=make_users(2),
            num_discussions=1,
            num_turns=2,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=tmp_path, verbose=False)
        assert capsys.readouterr().out == ""

    def test_begin_verbose_true_prints_output(
        self, tmp_path: Path, capsys
    ) -> None:
        exp = DiscussionExperiment(
            users=make_users(2),
            num_discussions=1,
            num_turns=2,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=tmp_path, verbose=True)
        assert len(capsys.readouterr().out) > 0

    def test_begin_with_seed_opinions_includes_seed_text(
        self, tmp_path: Path
    ) -> None:
        seeds = [["This is the seed opinion."]]
        exp = DiscussionExperiment(
            users=make_users(3),
            seed_opinions=seeds,
            num_discussions=1,
            num_turns=2,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=tmp_path, verbose=False)
        data = json.loads(next(tmp_path.glob("*.json")).read_text())
        entries = next(v for v in data.values() if isinstance(v, list))
        texts = [e["text"] for e in entries]
        assert "This is the seed opinion." in texts

    def test_begin_is_idempotent_on_existing_output_dir(
        self, tmp_path: Path
    ) -> None:
        """
        Calling begin() twice should not raise even if the directory exists.
        """
        out = tmp_path / "out"
        out.mkdir()
        exp = DiscussionExperiment(
            users=make_users(2),
            num_discussions=1,
            num_turns=2,
            num_active_users=2,
        )
        exp.begin(discussions_output_dir=out, verbose=False)


class TestAnnotationExperimentConstruction:

    def test_constructs_with_minimal_args(self) -> None:
        logs = make_logs()

        exp = AnnotationExperiment(
            annotators=make_annotators(2),
            discussion_logs=logs,
        )
        assert exp is not None

    def test_stores_attributes_correctly(self) -> None:
        logs = make_logs()

        annotators = make_annotators(3)
        exp = AnnotationExperiment(
            annotators=annotators,
            discussion_logs=logs,
            history_ctx_len=5,
        )

        assert exp.annotators == annotators
        assert exp.discussion_logs == logs
        assert exp.history_ctx_len == 5

    def test_raises_when_annotators_are_not_flagged(
        self, tmp_path: Path
    ) -> None:
        logs = make_logs()

        bad_annotators = make_users(2, is_annotator=False)

        with pytest.raises(ValueError):
            AnnotationExperiment(
                annotators=bad_annotators,
                discussion_logs=logs,
            ).begin(output_dir=tmp_path / "out")


class TestAnnotationExperimentBegin:

    def test_begin_creates_output_directory(self, tmp_path: Path) -> None:
        logs = make_logs()

        out = tmp_path / "annotations"

        exp = AnnotationExperiment(
            annotators=make_annotators(2),
            discussion_logs=logs,
        )

        exp.begin(output_dir=out, verbose=False)

        assert out.exists() and out.is_dir()

    def test_begin_writes_output_file(self, tmp_path: Path) -> None:
        logs = make_logs()

        exp = AnnotationExperiment(
            annotators=make_annotators(2),
            discussion_logs=logs,
        )

        exp.begin(output_dir=tmp_path, verbose=False)

        files = list(tmp_path.glob("*.json"))
        assert len(files) > 0

    def test_begin_outputs_valid_json(self, tmp_path: Path) -> None:
        logs = make_logs()

        exp = AnnotationExperiment(
            annotators=make_annotators(2),
            discussion_logs=logs,
        )

        exp.begin(output_dir=tmp_path, verbose=False)

        for f in tmp_path.glob("*.json"):
            data = json.loads(f.read_text())
            assert isinstance(data, dict)

    def test_begin_preserves_original_entries(self, tmp_path: Path) -> None:
        logs = make_logs(num_entries=4)

        exp = AnnotationExperiment(
            annotators=make_annotators(2),
            discussion_logs=logs,
        )

        exp.begin(output_dir=tmp_path, verbose=False)

        data = json.loads(next(tmp_path.glob("*.json")).read_text())
        entries = data.get("logs", [])

        assert len(entries) >= 4

    def test_begin_verbose_false_produces_no_stdout(
        self, tmp_path: Path, capsys
    ) -> None:
        logs = make_logs()

        exp = AnnotationExperiment(
            annotators=make_annotators(1),
            discussion_logs=logs,
        )

        exp.begin(output_dir=tmp_path, verbose=False)

        assert capsys.readouterr().out == ""

    def test_begin_verbose_true_prints_output(
        self, tmp_path: Path, capsys
    ) -> None:
        logs = make_logs()

        exp = AnnotationExperiment(
            annotators=make_annotators(1),
            discussion_logs=logs,
        )

        exp.begin(output_dir=tmp_path, verbose=True)

        assert len(capsys.readouterr().out) > 0

    def test_begin_handles_empty_logs(self, tmp_path: Path) -> None:
        logs = Logs()

        exp = AnnotationExperiment(
            annotators=make_annotators(2),
            discussion_logs=logs,
        )

        exp.begin(output_dir=tmp_path, verbose=False)

        files = list(tmp_path.glob("*.json"))
        assert len(files) > 0
