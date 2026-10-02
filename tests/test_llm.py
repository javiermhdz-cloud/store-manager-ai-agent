import json
from types import SimpleNamespace

import pytest

import src.llm as llm


@pytest.fixture(autouse=True)
def _no_fallback_model(monkeypatch):
    monkeypatch.delenv("LLM_FALLBACK_MODEL", raising=False)

class FakeModels:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


class FakeRateLimit(Exception):
    code = 429


def _response():
    return SimpleNamespace(
        text="Hola",
        usage_metadata=SimpleNamespace(
            prompt_token_count=4,
            candidates_token_count=2,
            thoughts_token_count=1,
        ),
    )


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    client = FakeClient([_response()])
    monkeypatch.setattr(llm.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(llm, "USAGE_LOG_PATH", tmp_path / "usage.jsonl")
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setattr(llm, "_last_call_started", None)
    monkeypatch.setattr(llm, "_wait_for_call_slot", lambda: None)
    return client


def test_generate_passes_tools_and_system_instruction_and_logs_only_usage(fake_client):
    response, usage = llm.generate(
        [{"role": "user", "parts": [{"text": "private prompt"}]}],
        tools=[{"function_declarations": [{"name": "lookup"}]}],
        system_instruction="Be accurate",
        interaction_type="datos",
    )

    assert response.text == "Hola"
    assert usage["input_tokens"] == 4
    assert usage["output_tokens"] == 2
    assert usage["thinking_tokens"] == 1
    call = fake_client.models.calls[0]
    assert call["model"] == "test-model"
    assert call["config"]["thinking_config"] == {"thinking_level": "LOW"}
    assert call["config"]["tools"] == [{"function_declarations": [{"name": "lookup"}]}]
    assert call["config"]["system_instruction"] == "Be accurate"
    logged = json.loads(llm.USAGE_LOG_PATH.read_text(encoding="utf-8"))
    assert logged == usage
    assert "private prompt" not in llm.USAGE_LOG_PATH.read_text(encoding="utf-8")
    assert "test-secret" not in llm.USAGE_LOG_PATH.read_text(encoding="utf-8")


def test_gemma_omits_thinking_config_but_keeps_tools_system_instruction_and_usage(
    fake_client, monkeypatch
):
    monkeypatch.setenv("LLM_MODEL", "gemma-3-27b-it")

    _, usage = llm.generate(
        "Hola",
        tools=[{"function_declarations": [{"name": "lookup"}]}],
        system_instruction="Be accurate",
        interaction_type="datos",
    )

    config = fake_client.models.calls[0]["config"]
    assert "thinking_config" not in config
    assert config["tools"] == [{"function_declarations": [{"name": "lookup"}]}]
    assert config["system_instruction"] == "Be accurate"
    assert usage["input_tokens"] == 4
    assert usage["output_tokens"] == 2
    assert json.loads(llm.USAGE_LOG_PATH.read_text(encoding="utf-8")) == usage


def test_thinking_level_uses_environment_override(fake_client, monkeypatch):
    monkeypatch.setenv("LLM_THINKING_LEVEL", "medium")

    llm.generate("Hola")

    assert fake_client.models.calls[0]["config"]["thinking_config"] == {
        "thinking_level": "MEDIUM"
    }


def test_generate_retries_rate_limits_and_logs_each_attempt(fake_client, monkeypatch):
    fake_client.models.responses = iter([FakeRateLimit(), _response()])
    delays = []
    monkeypatch.setattr(llm.time, "sleep", delays.append)

    _, usage = llm.generate("Hola", interaction_type="politica")

    assert len(fake_client.models.calls) == 2
    assert len(delays) == 1
    assert 2 <= delays[0] < 3
    records = [json.loads(line) for line in llm.USAGE_LOG_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["input_tokens"] == records[0]["output_tokens"] == 0
    assert records[1] == usage


def test_generate_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY is not set"):
        llm.generate("Hola")


def test_summarize_usage_averages_tokens_by_interaction_type(tmp_path):
    path = tmp_path / "usage.jsonl"
    path.write_text(
        '\n'.join(
            [
                json.dumps({"interaction_type": "datos", "input_tokens": 6, "output_tokens": 2, "thinking_tokens": 4}),
                json.dumps({"interaction_type": "datos", "input_tokens": 2, "output_tokens": 4}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert llm.summarize_usage(path) == {
        "datos": {
            "calls": 2,
            "average_input_tokens": 4.0,
            "average_output_tokens": 3.0,
            "average_thinking_tokens": 4.0,
        }
    }

def test_rate_limit_retries_at_most_five_attempts(fake_client, monkeypatch):
    error = FakeRateLimit()
    fake_client.models.responses = iter([error] * 5)
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    with pytest.raises(FakeRateLimit):
        llm.generate("Hola")

    assert len(fake_client.models.calls) == 5
    
class FakeServerError(Exception):
    def __init__(self, code):
        super().__init__(f"server error {code}")
        self.code = code


def test_server_errors_are_retried(fake_client, monkeypatch):
    fake_client.models.responses = iter([FakeServerError(503), FakeServerError(500), _response()])
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    llm.generate("Hola")

    assert len(fake_client.models.calls) == 3


def test_client_errors_are_not_retried(fake_client, monkeypatch):
    fake_client.models.responses = iter([FakeServerError(400), _response()])
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    with pytest.raises(FakeServerError):
        llm.generate("Hola")

    assert len(fake_client.models.calls) == 1


def test_fallback_model_used_after_retries(fake_client, monkeypatch):
    fake_client.models.responses = iter([FakeServerError(503)] * 5 + [_response()])
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    monkeypatch.setenv("LLM_FALLBACK_MODEL", "gemma-4-31b-it")

    llm.generate("Hola")

    assert len(fake_client.models.calls) == 6
    last = fake_client.models.calls[-1]
    assert last["model"] == "gemma-4-31b-it"
    assert "thinking_config" not in (last["config"] or {})