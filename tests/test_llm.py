import json
from types import SimpleNamespace

import pytest

import src.llm as llm


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
    assert call["config"]["tools"] == [{"function_declarations": [{"name": "lookup"}]}]
    assert call["config"]["system_instruction"] == "Be accurate"
    logged = json.loads(llm.USAGE_LOG_PATH.read_text(encoding="utf-8"))
    assert logged == usage
    assert "private prompt" not in llm.USAGE_LOG_PATH.read_text(encoding="utf-8")
    assert "test-secret" not in llm.USAGE_LOG_PATH.read_text(encoding="utf-8")


def test_generate_retries_rate_limits_and_logs_each_attempt(fake_client, monkeypatch):
    fake_client.models.responses = iter([FakeRateLimit(), _response()])
    delays = []
    monkeypatch.setattr(llm.time, "sleep", delays.append)

    _, usage = llm.generate("Hola", interaction_type="politica")

    assert len(fake_client.models.calls) == 2
    assert delays == [1]
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


def test_rate_limit_retries_at_most_three_times(fake_client, monkeypatch):
    error = FakeRateLimit()
    fake_client.models.responses = iter([error, error, error, error])
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)

    with pytest.raises(FakeRateLimit):
        llm.generate("Hola")

    assert len(fake_client.models.calls) == 4
    records = llm.USAGE_LOG_PATH.read_text(encoding="utf-8").splitlines()
    assert len(records) == 4