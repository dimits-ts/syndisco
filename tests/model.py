from syndisco.model import BaseModel


class DummyModel(BaseModel):
    """
    A deterministic test double for BaseModel.
    Returns a configurable fixed response and records every call made to it.
    """

    def __init__(
        self,
        name: str = "dummy",
        max_out_tokens: int = 256,
        stop_list: list[str] | None = None,
        fixed_response: str = "dummy response",
    ):
        super().__init__(name, max_out_tokens, stop_list)
        self.fixed_response = fixed_response
        self.calls: list[dict] = (
            []
        )  # stores (system_prompt, user_prompt) pairs

    def _generate_response(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({"system": system_prompt, "user": user_prompt})
        return self.fixed_response
