from syndisco.model import BaseModel


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
