"""
Unit tests for OpenAIModel.

No network access is needed for the default run:
- the OpenAI client class is replaced by a mock, or
- the real client talks to a throwaway HTTP server on localhost.

Live tests against a real endpoint are marked ``live`` and only run when
SYNDISCO_LIVE_BASE_URL and SYNDISCO_LIVE_MODEL are set.

The OpenAI import inside ``syndisco.model`` is lazy, so the mock is installed
on the ``openai`` module itself. Both ``import openai`` and
``from openai import OpenAI`` executed at call time pick it up.
"""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")

from openai.types.chat import ChatCompletion  # noqa: E402

from syndisco.model import OpenAIModel  # noqa: E402


MODEL_NAME = "test-model"
API_KEY = "sk-test"
BASE_URL = "http://fake-server.local/v1"


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def completion_payload(content: str | None) -> dict:
    """A minimal, valid chat-completion JSON body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def make_completion(content: str | None) -> ChatCompletion:
    """A real SDK response object, so attribute shapes match production."""
    return ChatCompletion.model_validate(completion_payload(content))


def namespace_response(**message_fields) -> SimpleNamespace:
    """
    A hand-built response for shapes the SDK types refuse to build.
    SimpleNamespace is used instead of MagicMock on purpose: MagicMock
    answers True to every hasattr() and would hide the missing-field branches.
    """
    message = SimpleNamespace(**message_fields)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def mock_openai_cls(monkeypatch):
    client_cls = MagicMock(name="OpenAI")
    monkeypatch.setattr(openai, "OpenAI", client_cls)
    return client_cls


@pytest.fixture
def create(mock_openai_cls):
    """The mocked chat.completions.create, returning 'hello' by default."""
    create_mock = mock_openai_cls.return_value.chat.completions.create
    create_mock.return_value = make_completion("hello")
    return create_mock


@pytest.fixture
def make_model(mock_openai_cls):
    def _make(**overrides) -> OpenAIModel:
        kwargs = {
            "model_name": MODEL_NAME,
            "api_key": API_KEY,
            "base_url": BASE_URL,
            "name": "tester",
            "max_out_tokens": 16,
        }
        kwargs.update(overrides)
        return OpenAIModel(**kwargs)

    return _make


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_client_is_built_with_credentials_and_base_url(
    make_model, mock_openai_cls
):
    model = make_model()

    mock_openai_cls.assert_called_once_with(api_key=API_KEY, base_url=BASE_URL)
    assert model.client is mock_openai_cls.return_value


def test_attributes_are_stored(make_model):
    model = make_model(temperature=0.7, max_out_tokens=99, name="alias")

    assert model.model_name == MODEL_NAME
    assert model.temperature == 0.7
    assert model.max_out_tokens == 99
    assert model.get_name() == "alias"


def test_default_temperature_is_zero(make_model):
    assert make_model().temperature == 0.0


def test_remove_string_list_becomes_stop_list(make_model):
    assert make_model(remove_string_list=["<eos>"]).stop_list == ["<eos>"]
    assert make_model().stop_list == []


def test_construction_does_not_call_the_api(make_model, create):
    make_model()
    create.assert_not_called()


def test_initialization_is_logged(make_model, caplog):
    caplog.set_level(logging.INFO)
    make_model()

    assert MODEL_NAME in caplog.text
    assert BASE_URL in caplog.text


# ---------------------------------------------------------------------------
# Request wiring
# ---------------------------------------------------------------------------


def test_request_arguments(make_model, create):
    model = make_model(temperature=0.3, max_out_tokens=32)
    model.prompt("system text", "user text")

    create.assert_called_once_with(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ],
        max_tokens=32,
        temperature=0.3,
    )


def test_one_request_per_prompt(make_model, create):
    model = make_model()
    model.prompt("s1", "u1")
    model.prompt("s2", "u2")

    assert create.call_count == 2
    sent_users = [c.kwargs["messages"][1]["content"] for c in create.call_args_list]
    assert sent_users == ["u1", "u2"]


def test_prompt_returns_stripped_content(make_model, create):
    create.return_value = make_completion("  an answer \n")
    assert make_model().prompt("s", "u") == "an answer"


# ---------------------------------------------------------------------------
# _validate_response
# ---------------------------------------------------------------------------


def test_validate_accepts_and_strips_valid_response(make_model):
    model = make_model()
    assert model._validate_response(make_completion("\n ok \t")) == "ok"


@pytest.mark.parametrize(
    ("response", "message_fragment"),
    [
        pytest.param(None, "empty response from OpenAI", id="none"),
        pytest.param(SimpleNamespace(), "'choices'", id="no-choices-attr"),
        pytest.param(SimpleNamespace(choices=[]), "'choices'", id="empty-choices"),
        pytest.param(
            SimpleNamespace(choices=[SimpleNamespace()]),
            "missing 'message'",
            id="no-message-attr",
        ),
        pytest.param(
            SimpleNamespace(choices=[SimpleNamespace(message=None)]),
            "missing 'message'",
            id="message-none",
        ),
        pytest.param(
            namespace_response(), "missing 'content'", id="no-content-attr"
        ),
        pytest.param(make_completion(None), "null content", id="content-none"),
        pytest.param(make_completion(""), "empty response", id="content-empty"),
        pytest.param(
            make_completion(" \n\t "), "empty response", id="content-whitespace"
        ),
    ],
)
def test_validate_rejects_malformed_response(make_model, response, message_fragment):
    with pytest.raises(ValueError, match=message_fragment):
        make_model()._validate_response(response)


@pytest.mark.parametrize("content", [123, ["a", "b"], {"text": "hi"}])
def test_validate_rejects_non_string_content(make_model, content):
    with pytest.raises(TypeError, match="expected str"):
        make_model()._validate_response(namespace_response(content=content))


def test_validation_errors_propagate_through_prompt(make_model, create):
    create.return_value = make_completion(None)
    with pytest.raises(ValueError, match="null content"):
        make_model().prompt("s", "u")


# ---------------------------------------------------------------------------
# Interaction with BaseModel's stop list
# ---------------------------------------------------------------------------


def test_stop_list_is_applied_after_stripping(make_model, create):
    # Documents current behaviour: the response is stripped *before*
    # stop strings are removed, so whitespace next to them survives.
    create.return_value = make_completion("  answer <eos>  ")
    model = make_model(remove_string_list=["<eos>"])

    assert model.prompt("s", "u") == "answer "


def test_stop_list_can_empty_a_validated_response(make_model, create):
    # Documents current behaviour: the emptiness check happens before
    # stop-string removal, so an all-stop-string reply returns "".
    create.return_value = make_completion("<eos>")
    model = make_model(remove_string_list=["<eos>"])

    assert model.prompt("s", "u") == ""


# ---------------------------------------------------------------------------
# SDK errors
# ---------------------------------------------------------------------------


def _request() -> "httpx.Request":
    return httpx.Request("POST", f"{BASE_URL}/chat/completions")


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            openai.APIConnectionError(request=_request()), id="connection"
        ),
        pytest.param(
            openai.RateLimitError(
                "rate limited",
                response=httpx.Response(429, request=_request()),
                body=None,
            ),
            id="rate-limit",
        ),
        pytest.param(
            openai.AuthenticationError(
                "bad key",
                response=httpx.Response(401, request=_request()),
                body=None,
            ),
            id="auth",
        ),
    ],
)
def test_sdk_errors_propagate(make_model, create, error):
    create.side_effect = error
    with pytest.raises(type(error)):
        make_model().prompt("s", "u")


# ---------------------------------------------------------------------------
# HTTP layer (real client, local fake server)
# ---------------------------------------------------------------------------
# A stdlib HTTP server on localhost is used instead of respx because recent
# SDK versions do not necessarily send requests through ``httpx``, which is
# what respx patches. A real socket works with any transport.


class _FakeServer:
    def __init__(self):
        self.requests: list[dict] = []
        self.status = 200
        self.body: dict = completion_payload("hi")

        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                fake.requests.append(
                    {
                        "path": self.path,
                        "headers": {k.lower(): v for k, v in self.headers.items()},
                        "json": json.loads(self.rfile.read(length) or b"{}"),
                    }
                )
                payload = json.dumps(fake.body).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):  # keep test output quiet
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )

    def respond(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body


@pytest.fixture
def fake_server():
    server = _FakeServer()
    server.thread.start()
    yield server
    server.httpd.shutdown()
    server.httpd.server_close()


def _real_model(base_url: str, **overrides) -> OpenAIModel:
    kwargs = {
        "model_name": MODEL_NAME,
        "api_key": API_KEY,
        "base_url": base_url,
        "name": "tester",
        "max_out_tokens": 8,
        "temperature": 0.5,
    }
    kwargs.update(overrides)
    return OpenAIModel(**kwargs)


def test_http_request_body_and_url(fake_server):
    fake_server.respond(200, completion_payload(" hi "))

    assert _real_model(fake_server.base_url).prompt("sys", "usr") == "hi"
    assert len(fake_server.requests) == 1

    request = fake_server.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"]["authorization"] == f"Bearer {API_KEY}"
    body = request["json"]
    assert body["model"] == MODEL_NAME
    assert body["max_tokens"] == 8
    assert body["temperature"] == 0.5
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


def test_http_null_content_is_rejected(fake_server):
    fake_server.respond(200, completion_payload(None))
    with pytest.raises(ValueError, match="null content"):
        _real_model(fake_server.base_url).prompt("s", "u")


def test_http_client_error_is_raised_without_retry(fake_server):
    # 4xx (other than 408/409/429) is not retried by the SDK, keeping this fast
    fake_server.respond(400, {"error": {"message": "bad request"}})
    with pytest.raises(openai.BadRequestError):
        _real_model(fake_server.base_url).prompt("s", "u")
    assert len(fake_server.requests) == 1


# ---------------------------------------------------------------------------
# Live smoke test (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_live_endpoint_smoke():
    base_url = os.environ.get("SYNDISCO_LIVE_BASE_URL")
    model_name = os.environ.get("SYNDISCO_LIVE_MODEL")
    if not (base_url and model_name):
        pytest.skip("Set SYNDISCO_LIVE_BASE_URL and SYNDISCO_LIVE_MODEL to run")

    model = OpenAIModel(
        model_name=model_name,
        api_key=os.environ.get("SYNDISCO_LIVE_API_KEY", "not-needed"),
        base_url=base_url,
        name="live",
        max_out_tokens=5,
    )
    response = model.prompt("Answer in one word.", "Say hello.")

    assert isinstance(response, str)
    assert response
