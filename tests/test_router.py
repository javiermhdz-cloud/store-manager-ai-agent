import json
from types import SimpleNamespace

import pytest

import src.router as router


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, **kwargs):
        self.calls.append((list(messages), kwargs))
        return next(self.responses), {
            "timestamp": str(len(self.calls)),
            "input_tokens": 10,
            "output_tokens": 4,
            "thinking_tokens": 2,
        }


def _response(content, *, function_calls=None, text="respuesta final"):
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        function_calls=function_calls or [],
        text=text,
    )


def _content(label):
    return SimpleNamespace(label=label, parts=[])


def _call(name, args=None, call_id="call-1"):
    return SimpleNamespace(name=name, args=args or {}, id=call_id)


@pytest.fixture
def router_logs(monkeypatch, tmp_path):
    interactions = tmp_path / "interactions.jsonl"
    errors = tmp_path / "router_errors.jsonl"
    monkeypatch.setattr(router, "INTERACTIONS_LOG_PATH", interactions)
    monkeypatch.setattr(router, "ROUTER_ERRORS_LOG_PATH", errors)
    return interactions, errors


def _set_fake_llm(monkeypatch, fake_llm):
    monkeypatch.setattr(router.llm, "generate", fake_llm.generate)
    monkeypatch.setattr(router.llm, "finalize_usage_records", lambda *_: None)


def _read_interaction(path):
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def test_tool_dispatch_and_original_response_content_are_preserved(
    monkeypatch, router_logs
):
    interactions, _ = router_logs
    called = []
    spec = router.TOOL_REGISTRY["sales_summary"]
    monkeypatch.setitem(
        router.TOOL_REGISTRY,
        "sales_summary",
        router.ToolSpec(lambda **kwargs: called.append(kwargs) or {"results": [{"unidades": 12}]}, spec.description, spec.parameters),
    )
    first_content = _content("thought-signature-bearing-original")
    final_content = _content("final")
    fake_llm = FakeLLM(
        [
            _response(
                first_content,
                function_calls=[_call("sales_summary", {"start_date": "2026-06-01", "end_date": "2026-06-01"})],
            ),
            _response(final_content, text="Se vendieron 12 unidades."),
        ]
    )
    _set_fake_llm(monkeypatch, fake_llm)

    result = router.answer("¿Cuántas unidades se vendieron?", history=["old"] * 7)

    assert result == "Se vendieron 12 unidades."
    assert called == [{"start_date": "2026-06-01", "end_date": "2026-06-01"}]
    assert fake_llm.calls[1][0][-2] is first_content
    assert fake_llm.calls[0][1]["tools"] == router.FUNCTION_TOOLS
    assert _read_interaction(interactions)["type"] == "datos"
    assert set(_read_interaction(interactions)) == {
        "type", "tool_names", "tool_calls", "input_tokens", "output_tokens", "thinking_tokens"
    }
    assert len(fake_llm.calls[0][0]) == 7


def test_tool_errors_are_returned_to_model_as_json(monkeypatch, router_logs):
    interactions, _ = router_logs
    spec = router.TOOL_REGISTRY["sales_summary"]
    monkeypatch.setitem(
        router.TOOL_REGISTRY,
        "sales_summary",
        router.ToolSpec(lambda **_: (_ for _ in ()).throw(ValueError("fecha fuera de rango")), spec.description, spec.parameters),
    )
    first_content = _content("call")
    fake_llm = FakeLLM(
        [
            _response(first_content, function_calls=[_call("sales_summary")]),
            _response(_content("final"), text="No tengo esa información. La fecha está fuera del rango disponible."),
        ]
    )
    _set_fake_llm(monkeypatch, fake_llm)

    assert "No tengo esa información" in router.answer("Consulta fuera del rango")

    tool_message = fake_llm.calls[1][0][-1]
    assert tool_message.parts[0].function_response.response == {"error": "fecha fuera de rango"}
    assert _read_interaction(interactions)["tool_calls"] == 1


def test_tool_loop_executes_at_most_five_rounds(monkeypatch, router_logs):
    interactions, _ = router_logs
    responses = [
        _response(_content(str(index)), function_calls=[_call("ticket_counts", {"group_by": "state"}, str(index))])
        for index in range(5)
    ]
    responses.append(_response(_content("final"), text="Resumen disponible."))
    fake_llm = FakeLLM(responses)
    _set_fake_llm(monkeypatch, fake_llm)

    result = router.answer("Resume los tickets")

    assert result == "Resumen disponible."
    assert len(fake_llm.calls) == 6
    assert fake_llm.calls[-1][1]["tools"] is None
    interaction = _read_interaction(interactions)
    assert interaction["tool_calls"] == 5
    assert interaction["tool_names"] == ["ticket_counts"]


def test_history_is_limited_to_last_six_messages(monkeypatch, router_logs):
    history = [f"history-{index}" for index in range(8)]
    fake_llm = FakeLLM([_response(_content("final"), text="No tengo esa información.")])
    _set_fake_llm(monkeypatch, fake_llm)

    router.answer("Hola", history=history)

    messages = fake_llm.calls[0][0]
    assert messages[:6] == history[-6:]
    assert len(messages) == 7


def test_system_prompt_includes_source_and_ticket_search_guidance(monkeypatch, router_logs):
    fake_llm = FakeLLM([_response(_content("final"), text="No tengo esa información.")])
    _set_fake_llm(monkeypatch, fake_llm)

    router.answer("¿Qué información falta?")

    system_instruction = fake_llm.calls[0][1]["system_instruction"]
    assert "Cuando una herramienta devuelva \"total\", úsalo como el número de elementos" in system_instruction
    assert "Si se usó un límite y total lo supera, indica que la lista está truncada." in system_instruction
    assert "total de 10 o menos, lista los elementos por nombre." in system_instruction
    assert "En preguntas de ventas, indica el periodo solicitado" in system_instruction
    assert "Para tickets, inventario y políticas, no menciones periodos, tiendas ni devoluciones salvo que la pregunta lo pida." in system_instruction
    assert "No menciones la fecha de corte del inventario salvo que la pregunta lo pida." in system_instruction
    assert "No agregues notas ni aclaraciones sobre herramientas, periodos o fuentes que no usaste." in system_instruction
    assert 'Cita los documentos con su nombre completo y acentos, por ejemplo "Política de mermas y caducidad".' in system_instruction
    assert "el conteo es por palabra clave" in system_instruction
    assert '"gotera" para fugas de agua' in system_instruction
    assert "terminan el 2026-08-31" in system_instruction
    assert "no hay pronósticos" in system_instruction
    assert "no hay datos de otras tiendas ni documentos fuera de los disponibles" in system_instruction


@pytest.mark.parametrize(
    ("question", "model_answer", "expected"),
    [
        (
            "¿Qué información falta?",
            "No tengo esa información. Falta el documento correspondiente.",
            "No tengo esa información. Falta el documento correspondiente.",
        ),
        (
            "¿Qué pasó el 2026-08-31?",
            "No tengo esa información para el 2026-08-31.",
            "No tengo esa información para el 2026-08-31.",
        ),
        (
            "¿Qué información falta?",
            "No tengo esa información. Hay 12 registros.",
            router._refusal(),
        ),
        (
            "¿Qué información falta?",
            "Necesitaría un documento adicional.",
            router._refusal(),
        ),
    ],
)
def test_no_tool_reply_preserves_only_grounded_model_refusals(
    monkeypatch, router_logs, question, model_answer, expected
):
    fake_llm = FakeLLM([_response(_content("final"), text=model_answer)])
    _set_fake_llm(monkeypatch, fake_llm)

    assert router.answer(question) == expected


def test_classification_combines_data_and_policy_tools(monkeypatch, router_logs):
    interactions, _ = router_logs
    fake_llm = FakeLLM(
        [
            _response(_content("data"), function_calls=[_call("ticket_counts", {"group_by": "state"})]),
            _response(_content("policy"), function_calls=[_call("search_policies", {"query": "devoluciones"})]),
            _response(_content("final"), text="La política aplica a estos tickets."),
        ]
    )
    _set_fake_llm(monkeypatch, fake_llm)

    router.answer("Combina datos y política")

    assert _read_interaction(interactions)["type"] == "ambos"


@pytest.mark.parametrize(
    ("function_calls", "expected_type"),
    [
        ([_call("search_policies", {"query": "devoluciones"})], "politica"),
        ([], "rechazo"),
    ],
)
def test_policy_and_rejection_classification(
    monkeypatch, router_logs, function_calls, expected_type
):
    interactions, _ = router_logs
    fake_llm = FakeLLM(
        [
            _response(
                _content("final"),
                function_calls=function_calls,
                text="No tengo esa información. Para responder necesitaría una fuente disponible.",
            ),
            *(
                [_response(_content("final answer"), text="No tengo esa información.")]
                if function_calls
                else []
            ),
        ]
    )
    _set_fake_llm(monkeypatch, fake_llm)

    router.answer("Consulta")

    assert _read_interaction(interactions)["type"] == expected_type


def test_grounding_retries_once_then_refuses_and_logs_failure(
    monkeypatch, router_logs
):
    interactions, errors = router_logs
    fake_llm = FakeLLM(
        [
            _response(_content("tool"), function_calls=[_call("ticket_counts", {"group_by": "state"})]),
            _response(_content("invented"), text="Hay 99 tickets."),
            _response(_content("still-invented"), text="Hay 98 tickets."),
        ]
    )
    monkeypatch.setitem(
        router.TOOL_REGISTRY,
        "ticket_counts",
        router.ToolSpec(lambda **_: {"results": [{"count": 12}]}, "Cuenta tickets", {}),
    )
    _set_fake_llm(monkeypatch, fake_llm)

    result = router.answer("¿Cuántos tickets hay?")

    assert result.startswith("No tengo esa información")
    assert len(fake_llm.calls) == 3
    assert json.loads(errors.read_text(encoding="utf-8")) == {
        "timestamp": json.loads(errors.read_text(encoding="utf-8"))["timestamp"],
        "event": "numeric_grounding_failed",
    }
    assert _read_interaction(interactions)["type"] == "datos"


def test_numeric_grounding_allows_values_from_the_question_or_tool_results():
    assert router._grounded("Se vendieron 1,430 unidades.", "¿Y 1,430?", [{"count": 1430.0}])
    assert not router._grounded("Se vendieron 99 unidades.", "¿Cuántas?", [{"count": 12}])
    assert not router._grounded("El saldo fue -5.", "¿Cuál fue el saldo?", [{"saldo": 5}])


def test_grounded_accepts_followup_with_iso_date_in_tool_result():
    tool_results = [{
        "filters": {"start_date": "2026-06-01", "end_date": "2026-06-01",
                    "store_id": "T02", "sku": "SKU-1041"},
        "results": [{"dia": "2026-06-01", "unidades": 22, "venta_neta_mxn": 682.0}],
    }]
    answer = ("El SKU-1041 en la tienda T02 vendió $682.00 MXN netos "
              "(22 unidades) el día 1 de junio de 2026.")
    assert router._grounded(answer, "¿y en la tienda T02?", tool_results)


def test_grounded_rejects_invented_number():
    tool_results = [{"filters": {"start_date": "2026-06-01"},
                     "results": [{"unidades": 22, "venta_neta_mxn": 682.0}]}]
    answer = "Vendió 23 unidades por $682.00 el 1 de junio de 2026."
    assert not router._grounded(answer, "pregunta", tool_results)


def test_grounded_rejects_wrong_date():
    tool_results = [{"filters": {"start_date": "2026-06-01"},
                     "results": [{"unidades": 22}]}]
    answer = "Vendió 22 unidades el 15 de junio de 2026."
    assert not router._grounded(answer, "pregunta", tool_results)

def test_grounded_rejects_iso_date_not_in_sources():
    answer = "El 2026-06-15 se vendieron 22 unidades."
    assert not router._grounded(answer, "pregunta", [{"unidades": 22}])


def test_grounded_accepts_iso_date_present_in_tool_result():
    tool_results = [{"dia": "2026-06-01", "unidades": 22}]
    answer = "El 2026-06-01 se vendieron 22 unidades."
    assert router._grounded(answer, "pregunta", tool_results)