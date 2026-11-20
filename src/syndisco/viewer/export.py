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
Export discussion logs as a single, self-contained SynDisco Viewer page.

The page contains the viewer's code, styles and the discussions themselves,
so it opens from disk in any browser, works offline and can be shared as
one file.
"""
import base64
import importlib.resources
import json
import re
import tempfile
import typing
import zipfile
from pathlib import Path

from .. import __version__


def static_dir() -> Path:
    """
    Location of the bundled viewer files.

    :return: the directory containing the viewer's ``index.html``.
    :rtype: Path
    """
    return Path(str(importlib.resources.files(__package__) / "static"))


def read_discussions(path: str | Path) -> list[dict[str, str]]:
    """
    Read the discussion files in a folder, JSON file or zip archive.

    Folders are searched recursively. Files and folders whose names start
    with a dot are skipped.

    :param path: the folder, ``.json`` file or ``.zip`` file to read.
    :type path: str | Path
    :raises FileNotFoundError: if *path* does not exist.
    :raises ValueError: if *path* is a file that is neither JSON nor zip,
        or contains no JSON files.
    :return: one ``{"path": ..., "text": ...}`` entry per file, with paths
        relative to *path* and sorted.
    :rtype: list[dict[str, str]]
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    files: dict[str, str] = {}
    if path.is_dir():
        for file in path.rglob("*.json"):
            rel = file.relative_to(path)
            if file.is_file() and not _is_hidden(rel.parts):
                files[rel.as_posix()] = file.read_text(encoding="utf8")
    elif path.suffix.lower() == ".json":
        files[path.name] = path.read_text(encoding="utf8")
    elif path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                parts = name.split("/")
                if name.lower().endswith(".json") and not _is_hidden(parts):
                    files[name] = archive.read(name).decode("utf8")
    else:
        raise ValueError(
            f"{path} must be a folder, a .json file or a .zip file"
        )

    if not files:
        raise ValueError(f"no JSON files found in {path}")
    return [{"path": rel, "text": files[rel]} for rel in sorted(files)]


def build_html(path: str | Path) -> str:
    """
    Build a self-contained viewer page with the discussions in *path*.

    :param path: the folder, ``.json`` file or ``.zip`` file to include.
    :type path: str | Path
    :return: the page's HTML.
    :rtype: str
    """
    static = static_dir()
    config = json.loads((static / "config.json").read_text(encoding="utf8"))
    config["version"] = __version__
    # the data is in the page, so there is nothing to remember between visits
    config["rememberData"] = False

    embedded = {
        "config": config,
        "label": Path(path).expanduser().resolve().name,
        "files": read_discussions(path),
    }
    # "<" is escaped so no file content can close the <script> element
    data_js = json.dumps(embedded).replace("<", "\\u003c")

    html = (static / "index.html").read_text(encoding="utf8")

    def inline_css(match: re.Match) -> str:
        css = (static / match[1]).read_text(encoding="utf8")
        return f"<style>\n{css}\n</style>"

    def inline_js(match: re.Match) -> str:
        js = (static / match[1]).read_text(encoding="utf8")
        return f"<script>\n{js}\n</script>"

    favicon = base64.b64encode((static / "favicon.svg").read_bytes()).decode()
    html = html.replace(
        'href="favicon.svg"', f'href="data:image/svg+xml;base64,{favicon}"'
    )
    html = re.sub(r'<link rel="stylesheet" href="([^"]+)">', inline_css, html)
    html = re.sub(r'<script src="([^"]+)" defer></script>', inline_js, html)
    # the data must be defined before the viewer's scripts run
    return html.replace(
        "</title>",
        f"</title>\n  <script>window.SYNDISCO_EMBEDDED = {data_js};</script>",
        1,
    )


def export(path: str | Path, output: str | Path) -> Path:
    """
    Write a self-contained viewer page with the discussions in *path*.

    :param path: the folder, ``.json`` file or ``.zip`` file to include.
    :type path: str | Path
    :param output: the HTML file to write; it is overwritten if it exists.
    :type output: str | Path
    :return: the written file.
    :rtype: Path
    """
    output = Path(output).expanduser()
    output.write_text(build_html(path), encoding="utf8")
    return output


def export_temporary(path: str | Path) -> Path:
    """
    Write the viewer page for *path* to a new temporary folder.

    The file is not deleted afterwards, since a browser may still need it;
    the operating system's temporary-file cleanup removes it eventually.

    :param path: the folder, ``.json`` file or ``.zip`` file to include.
    :type path: str | Path
    :return: the written file, named after *path*.
    :rtype: Path
    """
    html = build_html(path)  # fails before the folder is created
    folder = Path(tempfile.mkdtemp(prefix="syndisco-"))
    output = folder / (Path(path).expanduser().resolve().stem + ".html")
    output.write_text(html, encoding="utf8")
    return output


def _is_hidden(parts: typing.Sequence[str]) -> bool:
    return any(part.startswith(".") for part in parts)
