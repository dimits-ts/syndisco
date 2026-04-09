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
    ) -> None:
        self.name = name
        self._responses = responses or [f"{name} says something."]
        self._index = 0
        self.is_annotator = is_annotator
        self._model = DummyModel(["r1", "r2", "r3", "r4"])
        self.context = "context"
        self.instructions = "instructions"
        self.persona = {"persona": "exists"}
