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
from syndisco import BaseModel, Actor


class DummyModel(BaseModel):
    """
    Deterministic test double for BaseModel.

    Returns responses from *responses* in order.  Once exhausted the list
    wraps around (cycling), so a test that calls prompt() more times than
    there are canned responses never raises StopIteration.

    Example
    -------
    >>> m = DummyModel(["hello", "world"])
    >>> m.prompt("sys", "usr")
    'hello'
    >>> m.prompt("sys", "usr")
    'world'
    >>> m.prompt("sys", "usr")   # wraps
    'hello'
    """

    def __init__(self, responses: list[str]) -> None:
        super().__init__("dummy", 5, [])
        if not responses:
            raise ValueError("DummyModel requires at least one response.")
        self._responses = responses
        self._index = 0

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        response = self._responses[self._index % len(self._responses)]
        self._index += 1
        return response

    @property
    def call_count(self) -> int:
        """Total number of times prompt() has been called."""
        return self._index


class DummyActor(Actor):
    """Minimal Actor stub whose speak() cycles through preset strings."""

    def __init__(
        self,
        name: str = "Bot",
        responses: list[str] | None = None,
        is_annotator: bool = False,
        instructions: str = "",
    ) -> None:
        self.name = name
        self._responses = responses or [f"{name} says something."]
        self._index = 0
        self.is_annotator = is_annotator
        self._model = DummyModel(["r1", "r2", "r3", "r4"])
        self.context = "context"
        self.instructions = (
            "default" if instructions == "" else instructions
        )
        self.persona = {"persona": "exists"}
