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
Tests for the local server behind ``syndisco view``.

Only the standard library is needed; each test starts a server on a free
port in a background thread.
"""

import contextlib
import io
import json
import os
import socket
import threading
import urllib.error
import urllib.request
import zipfile

import pytest

from syndisco import __version__
from syndisco.viewer import server as viewer_server


def discussion(name="alice"):
    return json.dumps(
        {
            "timestamp": "26-07-01-10-00",
            "logs": [{"name": name, "text": "hi", "model": "m", "prompt": "p"}],
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
    outside = tmp_path / "outside.json"
    outside.write_text(discussion("outside"))
    with contextlib.suppress(OSError, NotImplementedError):
        os.symlink(outside, root / "link.json")
    return root


@contextlib.contextmanager
def running(path=None, **kwargs):
    srv = viewer_server.create_server(path=path, port=0, **kwargs)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def get(url, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as e:
        return e.code, b""


def test_serves_viewer_and_config(data_dir):
    with running(data_dir, show_prompts=True) as base:
        status, body = get(base + "/")
        assert status == 200 and b"SynDisco Viewer" in body
        config = json.loads(get(base + "/config.json")[1])
        assert config["showPrompts"] is True
        assert config["version"] == __version__


def test_config_defaults_hide_prompts():
    with running() as base:
        assert json.loads(get(base + "/config.json")[1])["showPrompts"] is False


def test_directory_listing_disabled():
    with running() as base:
        assert get(base + "/js/")[0] == 404


def test_datasets_index_autoloads_served_folder(data_dir):
    with running(data_dir) as base:
        index = json.loads(get(base + "/datasets/index.json")[1])
        assert index["autoload"] == "local"
        assert index["datasets"][0]["url"] == "local/data.zip"


def test_without_path_uses_static_datasets_index():
    with running() as base:
        index = json.loads(get(base + "/datasets/index.json")[1])
        assert "autoload" not in index
        assert get(base + "/local/data.zip")[0] == 404


def test_manifest_lists_only_visible_json_inside_folder(data_dir):
    with running(data_dir) as base:
        manifest = json.loads(get(base + "/local/manifest.json")[1])
        paths = [f["path"] for f in manifest["files"]]
        assert paths == ["exp1/a.json", "exp1/b.json", "top.json"]


def test_folder_is_served_as_zip(data_dir):
    with running(data_dir) as base:
        status, body = get(base + "/local/data.zip")
        assert status == 200
        names = sorted(zipfile.ZipFile(io.BytesIO(body)).namelist())
        assert names == ["exp1/a.json", "exp1/b.json", "top.json"]


def test_serves_files_inside_folder(data_dir):
    with running(data_dir) as base:
        status, body = get(base + "/local/files/exp1/a.json")
        assert status == 200
        assert json.loads(body)["logs"][0]["name"] == "a"


@pytest.mark.parametrize(
    "path",
    [
        "/local/files/../outside.json",
        "/local/files/..%2Foutside.json",
        "/local/files/%2e%2e/outside.json",
        "/local/files/link.json",
        "/local/files/notes.txt",
        "/local/files/.hidden/secret.json",
        "/local/files//etc/passwd",
    ],
)
def test_refuses_files_outside_or_not_json(data_dir, path):
    with running(data_dir) as base:
        assert get(base + path)[0] == 404


def test_rejects_foreign_host_header(data_dir):
    with running(data_dir) as base:
        status, _ = get(base + "/local/data.zip", headers={"Host": "evil.example"})
        assert status == 403
        port = base.rsplit(":", 1)[1]
        status, _ = get(base + "/", headers={"Host": f"localhost:{port}"})
        assert status == 200


def test_single_json_source(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(discussion())
    with running(path) as base:
        body = get(base + "/local/data.zip")[1]
        assert zipfile.ZipFile(io.BytesIO(body)).namelist() == ["one.json"]


def test_zip_source_is_served_unchanged(tmp_path):
    path = tmp_path / "data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("x/one.json", discussion())
    with running(path) as base:
        assert get(base + "/local/data.zip")[1] == path.read_bytes()


def test_invalid_paths(tmp_path):
    with pytest.raises(FileNotFoundError):
        viewer_server.DataSource(tmp_path / "missing")
    other = tmp_path / "data.csv"
    other.write_text("a,b")
    with pytest.raises(ValueError):
        viewer_server.DataSource(other)


def test_falls_back_to_next_free_port():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        srv = viewer_server.create_server(port=port)
        try:
            assert srv.server_address[1] != port
        finally:
            srv.server_close()


def test_static_files_are_packaged():
    static = viewer_server.static_dir()
    for name in (
        "index.html",
        "config.json",
        "datasets/index.json",
        "js/app.js",
        "js/vendor/fflate.min.js",
        "js/vendor/LICENSE-fflate.txt",
    ):
        assert (static / name).is_file(), name
