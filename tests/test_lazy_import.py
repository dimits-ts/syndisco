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
Tests for lazy imports in the package root.

``import syndisco`` must not load torch, transformers or openai, so that
lightweight entry points such as ``syndisco view`` start quickly.
"""

import subprocess
import sys

import pytest

import syndisco

HEAVY_MODULES = ("torch", "transformers", "openai")


def test_import_does_not_load_heavy_dependencies():
    code = (
        "import sys, syndisco, syndisco.cli, syndisco.viewer.server; "
        f"print([m for m in {HEAVY_MODULES!r} if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize("name", syndisco.__all__)
def test_public_names_resolve(name):
    assert getattr(syndisco, name) is not None


def test_from_import_still_works():
    from syndisco import Logs, Discussion  # noqa: F401


def test_dir_lists_public_names():
    assert set(syndisco.__all__) <= set(dir(syndisco))


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        syndisco.DoesNotExist  # noqa: B018
