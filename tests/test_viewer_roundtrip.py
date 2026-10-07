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
Round-trip tests between SynDisco and SynDisco Viewer.

Files downloaded from the viewer must load with ``Logs.from_file``. The
JavaScript tests run the viewer's own parsing and export code with Node.js
and are skipped when Node.js is not installed.
"""

import json
import shutil
import subprocess

import pytest

from syndisco import Logs
from syndisco.viewer.server import static_dir

NODE = shutil.which("node")

# Loads the viewer's scripts into a sandbox and runs one command on a file.
NODE_HARNESS = r"""
const fs = require("fs"), vm = require("vm"), path = require("path");
const [js, mode, input, output] = process.argv.slice(1);
const sandbox = {};
sandbox.self = sandbox;
vm.createContext(sandbox);
for (const f of ["strings.js", "data.js"]) {
  vm.runInContext(fs.readFileSync(path.join(js, f), "utf8"), sandbox);
}
const text = fs.readFileSync(input, "utf8");
const parsed = sandbox.SDVData.parseDiscussion(path.basename(input), text);
if (parsed.error) { console.error(parsed.error); process.exit(3); }
const r = parsed.record;
if (mode === "summary") {
  fs.writeFileSync(output, JSON.stringify({
    messages: r.messages.length, speakers: r.speakers, models: r.models,
    meta: r.meta, date: r.date && r.date.toISOString(), warnings: r.warnings,
  }));
} else {
  fs.writeFileSync(output, sandbox.SDVData.exportText(r, mode === "strip"));
}
"""


def make_logs():
    logs = Logs()
    logs.append(name="seed", text="Opening comment", model="hardcoded")
    logs.append(name="alice", text="Hello", model="m1", prompt="Be Alice.")
    logs.append(name="bob", text="Hi\nthere", model="m1", prompt="Be Bob.")
    return logs


def test_from_file_accepts_viewer_extensions(tmp_path):
    """Extra keys and empty prompts, as the viewer may export, still load."""
    data = make_logs().to_dict()
    data["experiment"] = "exp-1"
    for entry in data["logs"]:
        entry["prompt"] = ""
        entry["score"] = 0.5
    path = tmp_path / "extended.json"
    path.write_text(json.dumps(data, indent=4))
    loaded = Logs.from_file(path)
    assert [e["text"] for e in loaded] == [e["text"] for e in make_logs()]


def run_node(tmp_path, mode, source):
    output = tmp_path / f"out-{mode}.json"
    subprocess.run(
        [NODE, "-e", NODE_HARNESS, str(static_dir() / "js"), mode, str(source), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_viewer_reads_syndisco_export(tmp_path):
    source = tmp_path / "26-07-01-10-00-00.json"
    make_logs().export(source)
    summary = json.loads(run_node(tmp_path, "summary", source).read_text())
    assert summary["messages"] == 3
    assert summary["speakers"] == ["seed", "alice", "bob"]
    assert summary["models"] == ["m1"]
    assert summary["date"] is not None
    assert summary["warnings"] == []


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_viewer_export_is_unchanged_and_reloads(tmp_path):
    source = tmp_path / "d.json"
    make_logs().export(source)
    exported = run_node(tmp_path, "raw", source)
    assert exported.read_bytes() == source.read_bytes()
    assert Logs.from_file(exported) == Logs.from_file(source)


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_viewer_strip_prompts_export_reloads(tmp_path):
    source = tmp_path / "d.json"
    make_logs().export(source)
    data = json.loads(source.read_text())
    data["experiment"] = "kept"
    source.write_text(json.dumps(data, indent=4))

    stripped = json.loads(run_node(tmp_path, "strip", source).read_text())
    assert all(entry["prompt"] == "" for entry in stripped["logs"])
    assert stripped["experiment"] == "kept"
    assert stripped["timestamp"] == data["timestamp"]

    path = tmp_path / "stripped.json"
    path.write_text(json.dumps(stripped))
    assert Logs.from_file(path) == Logs.from_file(source)
