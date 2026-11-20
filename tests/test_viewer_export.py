# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2026 Dimitris Tsirmpas

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
Tests for ``syndisco view`` and ``syndisco export``, which write a
self-contained viewer page.
"""

import json
import re
import zipfile
from pathlib import Path

import pytest

from syndisco import __version__, cli
from syndisco.viewer import export


def discussion(name="alice"):
    return json.dumps(
        {
            "timestamp": "26-07-01-10-00",
            "logs": [
                {"name": name, "text": "hi", "model": "m", "prompt": "p"}
            ],
        }
    )


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "logs"
    (root / "exp1").mkdir(parents=True)
    (root / "exp1" / "a.json").write_text(discussion("a"))
    (root / "exp1" / "b.json").write_text(discussion("b"))
    (root / "top.json").write_text(discussion("c"))
    (root / "notes.txt").write_text("not a discussion")
    (root / ".hidden").mkdir()
    (root / ".hidden" / "secret.json").write_text(discussion("hidden"))
    return root


def embedded(html):
    """The data a page passes to the viewer."""
    match = re.search(r"window\.SYNDISCO_EMBEDDED = (.*?);</script>", html)
    assert match, "no embedded data in the page"
    return json.loads(match[1])


# -- reading discussions -------------------------------------------------


def test_reads_folder_recursively_skipping_hidden_and_non_json(data_dir):
    files = export.read_discussions(data_dir)
    assert [f["path"] for f in files] == [
        "exp1/a.json",
        "exp1/b.json",
        "top.json",
    ]
    assert files[0]["text"] == discussion("a")


def test_reads_single_json_file(data_dir):
    files = export.read_discussions(data_dir / "top.json")
    assert files == [{"path": "top.json", "text": discussion("c")}]


def test_reads_zip(tmp_path):
    archive = tmp_path / "logs.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("exp/a.json", discussion("a"))
        z.writestr("readme.txt", "ignored")
        z.writestr("__MACOSX/._a.json", "ignored")
        z.writestr("exp/.b.json", "ignored")
    files = export.read_discussions(archive)
    assert [f["path"] for f in files] == ["exp/a.json"]


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        export.read_discussions(tmp_path / "nope")


def test_unsupported_file_raises(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("x")
    with pytest.raises(ValueError):
        export.read_discussions(other)


def test_folder_without_json_raises(tmp_path):
    with pytest.raises(ValueError):
        export.read_discussions(tmp_path)


# -- building the page ---------------------------------------------------


def test_page_is_self_contained(data_dir):
    html = export.build_html(data_dir)
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert 'href="favicon.svg"' not in html
    assert "SDVSearch" in html  # search-core.js was inlined


def test_page_embeds_data_and_config(data_dir):
    data = embedded(export.build_html(data_dir))
    assert data["label"] == "logs"
    assert [f["path"] for f in data["files"]] == [
        "exp1/a.json",
        "exp1/b.json",
        "top.json",
    ]
    assert data["config"]["version"] == __version__
    assert data["config"]["rememberData"] is False


def test_data_is_defined_before_viewer_scripts(data_dir):
    html = export.build_html(data_dir)
    assert html.index("SYNDISCO_EMBEDDED =") < html.index("<script>\n")


def test_file_contents_cannot_close_the_script(tmp_path):
    hostile = tmp_path / "x.json"
    text = json.dumps({"logs": [{"name": "a", "text": "</script><b>"}]})
    hostile.write_text(text)
    html = export.build_html(hostile)
    assert "</script><b>" not in html
    assert embedded(html)["files"][0]["text"] == text


# -- temporary pages -----------------------------------------------------


def test_temporary_page_is_named_after_source(data_dir):
    page = export.export_temporary(data_dir)
    assert page.name == "logs.html"
    assert page.parent.name.startswith("syndisco-")
    assert embedded(page.read_text())["label"] == "logs"


def test_temporary_pages_do_not_collide(data_dir):
    first = export.export_temporary(data_dir)
    second = export.export_temporary(data_dir)
    assert first.parent != second.parent
    assert first.is_file() and second.is_file()


def test_temporary_page_errors_create_no_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(export.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        export.export_temporary(tmp_path / "nope")
    assert list(tmp_path.iterdir()) == []


# -- command line ----------------------------------------------------------


@pytest.fixture
def opened(monkeypatch):
    """URLs the CLI asked the browser to open."""
    urls = []
    monkeypatch.setattr(cli.webbrowser, "open", urls.append)
    return urls


def test_export_writes_target_without_opening(data_dir, tmp_path, opened):
    target = tmp_path / "page.html"
    assert cli.main(["export", str(data_dir), str(target)]) == 0
    assert embedded(target.read_text())["label"] == "logs"
    assert opened == []


def test_export_overwrites_target(data_dir, tmp_path, opened):
    target = tmp_path / "page.html"
    target.write_text("old")
    assert cli.main(["export", str(data_dir), str(target)]) == 0
    assert target.read_text() != "old"


def test_export_prints_target(data_dir, tmp_path, opened, capsys):
    target = tmp_path / "page.html"
    cli.main(["export", str(data_dir), str(target)])
    assert str(target) in capsys.readouterr().out


def test_export_requires_target(data_dir):
    with pytest.raises(SystemExit) as exc:
        cli.main(["export", str(data_dir)])
    assert exc.value.code == 2


def test_view_opens_temporary_page(data_dir, opened, capsys):
    assert cli.main(["view", str(data_dir)]) == 0
    assert len(opened) == 1
    page = Path(capsys.readouterr().out.split("Wrote ", 1)[1].strip())
    assert page.name == "logs.html"
    assert opened == [page.resolve().as_uri()]


@pytest.mark.parametrize("option", ["-o", "--no-browser", "--show-prompts"])
def test_view_has_no_options(data_dir, option):
    with pytest.raises(SystemExit) as exc:
        cli.main(["view", str(data_dir), option])
    assert exc.value.code == 2


@pytest.mark.parametrize("command", ["view", "export"])
def test_errors_are_reported(command, tmp_path, opened, capsys):
    args = [command, str(tmp_path / "nope")]
    if command == "export":
        args.append(str(tmp_path / "page.html"))
    assert cli.main(args) == 2
    assert f"syndisco {command}:" in capsys.readouterr().err
    assert opened == []


def test_corrupt_zip_is_reported(tmp_path, capsys):
    broken = tmp_path / "broken.zip"
    broken.write_bytes(b"not a zip")
    target = tmp_path / "page.html"
    assert cli.main(["export", str(broken), str(target)]) == 2
    assert "syndisco export:" in capsys.readouterr().err
    assert not target.exists()
