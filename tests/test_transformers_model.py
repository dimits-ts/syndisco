"""
Unit tests for TransformersModel.

Two tiers:
- Fast tests replace ``from_pretrained`` with mocks. Real (tiny) torch
  tensors are still used where the class does tensor indexing, so the
  prompt-slicing logic is actually exercised.
- ``slow`` tests build a tiny random-weight GPT-2 and a word-level tokenizer
  entirely offline, save them to a temp dir, and load them through the real
  code path. No download from the HuggingFace Hub is needed.

The transformers/torch imports inside ``syndisco.model`` are lazy, so mocks
are installed on the ``transformers`` classes themselves; attribute lookups
done at call time pick them up.
"""

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from syndisco.model import TransformersModel  # noqa: E402


MODEL_PATH = "fake/model-path"
EOS_ID = 99
PROMPT_IDS = [11, 12, 13]
NEW_IDS = [21, 22]


# ---------------------------------------------------------------------------
# Fast tier: fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_model():
    model = MagicMock(name="model")
    # The class keeps whatever .eval() returns, so make it return itself
    model.eval.return_value = model
    model.device = torch.device("cpu")
    model.get_memory_footprint.return_value = 5 * 2**20  # 5 MB
    model.generate.return_value = torch.tensor([PROMPT_IDS + NEW_IDS])
    return model


@pytest.fixture
def fake_tokenizer():
    tokenizer = MagicMock(name="tokenizer")
    tokenizer.apply_chat_template.return_value = "<templated prompt>"
    tokenizer.eos_token_id = EOS_ID
    # A real BatchEncoding: supports .to(device) and ** unpacking
    tokenizer.return_value = transformers.BatchEncoding(
        {
            "input_ids": torch.tensor([PROMPT_IDS]),
            "attention_mask": torch.ones(1, len(PROMPT_IDS), dtype=torch.long),
        }
    )
    tokenizer.decode.return_value = "  generated text \n"
    return tokenizer


@pytest.fixture
def loaders(monkeypatch, fake_model, fake_tokenizer):
    model_loader = MagicMock(name="AutoModelForCausalLM.from_pretrained")
    model_loader.return_value = fake_model
    tokenizer_loader = MagicMock(name="AutoTokenizer.from_pretrained")
    tokenizer_loader.return_value = fake_tokenizer

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", model_loader
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    return model_loader, tokenizer_loader


@pytest.fixture
def make_model(loaders):
    def _make(**overrides) -> TransformersModel:
        kwargs = {"model_path": MODEL_PATH, "name": "tester", "max_out_tokens": 16}
        kwargs.update(overrides)
        return TransformersModel(**kwargs)

    return _make


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_model_is_loaded_with_auto_device_map_and_kwargs(make_model, loaders):
    model_loader, _ = loaders
    make_model(model_kwargs={"dtype": "float32", "trust_remote_code": False})

    model_loader.assert_called_once_with(
        MODEL_PATH, device_map="auto", dtype="float32", trust_remote_code=False
    )


def test_tokenizer_is_loaded_with_kwargs(make_model, loaders):
    _, tokenizer_loader = loaders
    make_model(tokenizer_kwargs={"use_fast": True})

    tokenizer_loader.assert_called_once_with(MODEL_PATH, use_fast=True)


def test_path_objects_are_passed_through(make_model, loaders):
    model_loader, tokenizer_loader = loaders
    path = Path("some/dir")
    make_model(model_path=path)

    assert model_loader.call_args.args[0] == path
    assert tokenizer_loader.call_args.args[0] == path


def test_model_is_put_in_eval_mode(make_model, fake_model):
    model = make_model()

    fake_model.eval.assert_called_once_with()
    assert model.model is fake_model
    fake_model.generate.assert_not_called()


def test_kwarg_defaults(make_model, loaders):
    model_loader, tokenizer_loader = loaders
    model = make_model()

    model_loader.assert_called_once_with(MODEL_PATH, device_map="auto")
    tokenizer_loader.assert_called_once_with(MODEL_PATH)
    assert model.generation_kwargs == {}
    assert model.stop_list == []


def test_remove_string_list_becomes_stop_list(make_model):
    assert make_model(remove_string_list=["<eos>"]).stop_list == ["<eos>"]


def test_attributes_are_stored(make_model):
    model = make_model(name="alias", max_out_tokens=7)
    assert model.get_name() == "alias"
    assert model.max_out_tokens == 7


def test_memory_footprint_is_logged(make_model, caplog):
    caplog.set_level(logging.INFO)
    make_model()
    assert "5.00 MB" in caplog.text


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_chat_template_is_used_when_available(make_model, fake_tokenizer):
    make_model().prompt("system text", "user text")

    fake_tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "system", "content": "system text"},
            {"role": "user", "content": "user text"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    fake_tokenizer.assert_called_once_with("<templated prompt>", return_tensors="pt")


def test_fallback_prompt_without_apply_chat_template(
    make_model, fake_tokenizer, caplog
):
    del fake_tokenizer.apply_chat_template  # hasattr() now returns False
    caplog.set_level(logging.WARNING)

    make_model().prompt("sys", "usr")

    fake_tokenizer.assert_called_once_with(
        "System: sys\nUser: usr\nAssistant:", return_tensors="pt"
    )
    assert "no chat template" in caplog.text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def test_generate_arguments(make_model, fake_model):
    make_model(max_out_tokens=32).prompt("s", "u")

    kwargs = fake_model.generate.call_args.kwargs
    assert torch.equal(kwargs.pop("input_ids"), torch.tensor([PROMPT_IDS]))
    assert torch.equal(
        kwargs.pop("attention_mask"), torch.ones(1, len(PROMPT_IDS), dtype=torch.long)
    )
    assert kwargs == {
        "max_new_tokens": 32,
        "do_sample": False,
        "pad_token_id": EOS_ID,
    }


def test_generation_kwargs_are_forwarded(make_model, fake_model):
    make_model(
        generation_kwargs={"repetition_penalty": 1.2, "num_beams": 2}
    ).prompt("s", "u")

    kwargs = fake_model.generate.call_args.kwargs
    assert kwargs["repetition_penalty"] == 1.2
    assert kwargs["num_beams"] == 2


def test_generation_runs_in_inference_mode(make_model, fake_model):
    seen = {}

    def record_mode(**kwargs):
        seen["inference_mode"] = torch.is_inference_mode_enabled()
        return torch.tensor([PROMPT_IDS + NEW_IDS])

    fake_model.generate.side_effect = record_mode
    make_model().prompt("s", "u")

    assert seen["inference_mode"] is True


def test_only_new_tokens_are_decoded(make_model, fake_tokenizer):
    make_model().prompt("s", "u")

    decoded_ids = fake_tokenizer.decode.call_args.args[0]
    assert decoded_ids.tolist() == NEW_IDS
    assert fake_tokenizer.decode.call_args.kwargs == {"skip_special_tokens": True}


def test_no_new_tokens_decodes_empty_sequence(make_model, fake_model, fake_tokenizer):
    fake_model.generate.return_value = torch.tensor([PROMPT_IDS])
    fake_tokenizer.decode.return_value = ""

    assert make_model().prompt("s", "u") == ""
    assert fake_tokenizer.decode.call_args.args[0].tolist() == []


def test_response_is_stripped(make_model):
    assert make_model().prompt("s", "u") == "generated text"


def test_stop_list_is_applied_after_stripping(make_model, fake_tokenizer):
    # Documents current behaviour: strip happens in _generate_response,
    # stop-string removal afterwards in BaseModel.prompt.
    fake_tokenizer.decode.return_value = "  answer <eos>  "
    model = make_model(remove_string_list=["<eos>"])

    assert model.prompt("s", "u") == "answer "


def test_one_generate_call_per_prompt(make_model, fake_model):
    model = make_model()
    model.prompt("s1", "u1")
    model.prompt("s2", "u2")

    assert fake_model.generate.call_count == 2


# ---------------------------------------------------------------------------
# Known issues (strict xfail: they start failing loudly once fixed)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    raises=TypeError,
    strict=True,
    reason="generation_kwargs cannot override the hard-coded generate() "
    "arguments (do_sample, max_new_tokens, pad_token_id): Python raises "
    "'got multiple values for keyword argument'.",
)
@pytest.mark.parametrize(
    "override",
    [{"do_sample": True}, {"max_new_tokens": 4}, {"pad_token_id": 0}],
    ids=["do_sample", "max_new_tokens", "pad_token_id"],
)
def test_generation_kwargs_can_override_defaults(make_model, fake_model, override):
    make_model(generation_kwargs=override).prompt("s", "u")

    kwargs = fake_model.generate.call_args.kwargs
    for key, value in override.items():
        assert kwargs[key] == value


# ---------------------------------------------------------------------------
# Slow tier: tiny real model, built offline
# ---------------------------------------------------------------------------

VOCAB_WORDS = [
    "[UNK]", "[PAD]", "[EOS]",
    "system", "user", "assistant", ":",
    "the", "a", "cat", "dog", "sat", "on", "mat", "ran", "to",
    "hello", "world", "is", "big", "small", "red", "blue", "and",
]  # fmt: skip

CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} {% endfor %}"
    "{% if add_generation_prompt %}assistant : {% endif %}"
)

LONG_USER_PROMPT = " ".join(["the cat sat on the mat and the dog ran"] * 3)


def _build_tiny_checkpoint(directory: Path, with_chat_template: bool) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {word: i for i, word in enumerate(VOCAB_WORDS)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    if with_chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE

    config = transformers.GPT2Config(
        vocab_size=len(vocab),
        n_positions=128,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=vocab["[EOS]"],
        eos_token_id=vocab["[EOS]"],
        pad_token_id=vocab["[PAD]"],
    )
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(config)

    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


@pytest.fixture(scope="session")
def tiny_checkpoint(tmp_path_factory):
    pytest.importorskip("accelerate")  # needed for device_map="auto"
    return _build_tiny_checkpoint(
        tmp_path_factory.mktemp("tiny_chat"), with_chat_template=True
    )


@pytest.fixture(scope="session")
def tiny_checkpoint_no_template(tmp_path_factory):
    pytest.importorskip("accelerate")
    return _build_tiny_checkpoint(
        tmp_path_factory.mktemp("tiny_plain"), with_chat_template=False
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_checkpoint):
    return TransformersModel(tiny_checkpoint, name="tiny", max_out_tokens=5)


@pytest.mark.slow
def test_tiny_model_returns_string(tiny_model):
    assert isinstance(tiny_model.prompt("the cat", "hello world"), str)


@pytest.mark.slow
def test_tiny_model_respects_max_out_tokens_and_drops_prompt(tiny_model):
    # The prompt is far longer than max_out_tokens, so if prompt tokens
    # leaked into the decoded output this bound would be exceeded.
    response = tiny_model.prompt("the cat", LONG_USER_PROMPT)
    n_tokens = len(tiny_model.tokenizer(response)["input_ids"])

    assert n_tokens <= tiny_model.max_out_tokens


@pytest.mark.slow
def test_tiny_model_is_deterministic(tiny_model):
    first = tiny_model.prompt("the cat", "the dog ran")
    second = tiny_model.prompt("the cat", "the dog ran")
    assert first == second


@pytest.mark.slow
def test_tiny_model_is_in_eval_mode(tiny_model):
    assert not tiny_model.model.training


@pytest.mark.slow
def test_tiny_model_accepts_generation_kwargs(tiny_checkpoint):
    model = TransformersModel(
        tiny_checkpoint,
        name="tiny",
        max_out_tokens=5,
        generation_kwargs={"repetition_penalty": 1.3, "no_repeat_ngram_size": 2},
    )
    assert isinstance(model.prompt("the cat", "the dog"), str)


@pytest.mark.slow
def test_tiny_model_applies_stop_list(tiny_checkpoint):
    plain = TransformersModel(tiny_checkpoint, name="tiny", max_out_tokens=5)
    words = plain.prompt("the cat", "the dog ran").split()
    if not words:
        pytest.skip("Random model produced no visible tokens for this prompt")

    filtered = TransformersModel(
        tiny_checkpoint,
        name="tiny",
        max_out_tokens=5,
        remove_string_list=[words[0]],
    )
    assert words[0] not in filtered.prompt("the cat", "the dog ran")


@pytest.mark.slow
@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="Tokenizers always have apply_chat_template(), so the hasattr() "
    "check never triggers the fallback; a tokenizer whose chat_template "
    "is None raises ValueError instead. Checking "
    "getattr(tokenizer, 'chat_template', None) would fix this.",
)
def test_tiny_model_falls_back_without_chat_template(tiny_checkpoint_no_template):
    model = TransformersModel(
        tiny_checkpoint_no_template, name="tiny", max_out_tokens=5
    )
    assert isinstance(model.prompt("the cat", "the dog"), str)
