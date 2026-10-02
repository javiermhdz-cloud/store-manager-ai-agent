"""Tool-calling assistant router for store data and internal policies."""

from __future__ import annotations

import csv
import inspect
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import UnionType
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints
import os

from google.genai import types

from src import data_tools, llm, policies
from src.config import DATA_DIR, INVENTORY_SNAPSHOT_DATE, SALES_END_DATE, SALES_START_DATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERACTIONS_LOG_PATH = PROJECT_ROOT / "logs" / "interactions.jsonl"
ROUTER_ERRORS_LOG_PATH = PROJECT_ROOT / "logs" / "router_errors.jsonl"
MAX_TOOL_ROUNDS = 5
DATA_TOOL_NAMES = {
    "sales_summary",
    "rank_skus",
    "sku_info",
    "stock_below_reorder",
    "lots_expiring_soon",
    "ticket_counts",
    "list_tickets",
    "search_tickets",
}


@dataclass(frozen=True)
class ToolSpec:
    function: Callable[..., Any]
    description: str
    parameters: dict[str, Any]


TOOL_DESCRIPTIONS = {
    "sales_summary": "Resume ventas netas y unidades por periodo, tienda, categoría o canal. Incluye devoluciones por defecto.",
    "rank_skus": "Ordena productos por ventas netas o unidades en el periodo solicitado; incluye devoluciones por defecto.",
    "sku_info": "Busca un producto por SKU o nombre e informa su inventario de corte y sustituto, si existe.",
    "stock_below_reorder": "Lista inventario del corte que está en o debajo del punto de reorden.",
    "lots_expiring_soon": "Lista lotes perecederos próximos a caducar desde la fecha de corte, incluidos los ya caducados.",
    "ticket_counts": "Cuenta tickets por categoría, prioridad, estado o tienda con filtros estructurados.",
    "list_tickets": "Lista tickets recientes con filtros estructurados.",
    "search_tickets": "Busca texto aproximado en tickets. Usa entre 1 y 3 palabras clave significativas.",
    "search_policies": "Busca secciones relevantes en los documentos de políticas proporcionados; cita solo resultados recuperados.",
}


def _store_ids() -> list[str]:
    with (DATA_DIR / "tiendas.csv").open(encoding="utf-8", newline="") as stores_file:
        return sorted(row["tienda_id"] for row in csv.DictReader(stores_file))


POLICY_DOCUMENTS = sorted(path.name for path in (PROJECT_ROOT / "politicas").glob("*.pdf"))
SYSTEM_PROMPT = f"""Eres un asistente para una gerente de tiendas. Responde en español de México, de forma breve y clara.

Reglas obligatorias:
- Toda cifra, conteo, monto, porcentaje, unidad y fecha factual debe venir de los resultados de las herramientas. No calcules, estimes ni recuerdes cifras. No restes descuentos: venta_neta_mxn ya es neta de descuentos.
- Cuando uses datos, indica siempre el periodo solicitado, la tienda o tiendas cubiertas y si se incluyeron devoluciones. Si no aplica (inventario o tickets), dilo explícitamente. Ventas cubren {SALES_START_DATE.isoformat()} a {SALES_END_DATE.isoformat()}; el inventario es solo una fotografía al {INVENTORY_SNAPSHOT_DATE.isoformat()}, nunca una serie histórica.
- Las tiendas disponibles son: {", ".join(_store_ids())}. Los documentos disponibles son: {", ".join(POLICY_DOCUMENTS)}.
- Responde sobre políticas solo con secciones recuperadas por search_policies. Cita cada afirmación de política con el documento, versión y sección en el formato exacto "(Documento vX.X, §N.N)". No inventes citas ni respondas desde conocimiento general.
- Para search_tickets usa de 1 a 3 palabras clave significativas. Si no hay resultados, intenta de nuevo con match_mode="any" o con palabras relacionadas.
- Rechaza pronósticos, historia de inventario, tiendas no listadas, fechas fuera del periodo de ventas para consultas de ventas, inventario fuera del corte indicado y políticas fuera de los documentos. La negativa debe incluir literalmente "No tengo esa información" y qué dato o documento haría falta.
- No respondas con cifras si no se respaldan con resultados de herramientas. Evita listas numeradas para no introducir cifras sin respaldo.
"""


def _json_type(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is Literal:
        values = get_args(annotation)
        if values and all(isinstance(value, str) for value in values):
            return "string"
        if values and all(isinstance(value, int) for value in values):
            return "integer"
        return "string"
    if origin in (Union, UnionType):
        options = [value for value in get_args(annotation) if value is not type(None)]
        return _json_type(options[0]) if options else "string"
    if annotation is bool:
        return "boolean"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    return "string"


def _parameters_for(function: Callable[..., Any]) -> dict[str, Any]:
    hints = get_type_hints(function)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in inspect.signature(function).parameters.items():
        if name not in hints:
            continue
        schema: dict[str, Any] = {"type": _json_type(hints[name])}
        if get_origin(hints[name]) is Literal:
            schema["enum"] = list(get_args(hints[name]))
        properties[name] = schema
        if parameter.default is parameter.empty:
            required.append(name)
    parameters: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        parameters["required"] = required
    return parameters


def _tool_spec(function: Callable[..., Any], name: str) -> ToolSpec:
    return ToolSpec(function, TOOL_DESCRIPTIONS[name], _parameters_for(function))


TOOL_REGISTRY: dict[str, ToolSpec] = {
    name: _tool_spec(function, name)
    for name, function in (
        ("sales_summary", data_tools.sales_summary),
        ("rank_skus", data_tools.rank_skus),
        ("sku_info", data_tools.sku_info),
        ("stock_below_reorder", data_tools.stock_below_reorder),
        ("lots_expiring_soon", data_tools.lots_expiring_soon),
        ("ticket_counts", data_tools.ticket_counts),
        ("list_tickets", data_tools.list_tickets),
        ("search_tickets", data_tools.search_tickets),
        ("search_policies", policies.search_policies),
    )
}
FUNCTION_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name=name,
                description=spec.description,
                parameters_json_schema=spec.parameters,
            )
            for name, spec in TOOL_REGISTRY.items()
        ]
    )
]


def _content(response: Any) -> Any:
    candidates = getattr(response, "candidates", None) or []
    return candidates[0].content if candidates else None


def _function_calls(response: Any) -> list[Any]:
    return list(getattr(response, "function_calls", None) or [])


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if text is not None:
        return str(text)
    content = _content(response)
    return "".join(
        str(part.text)
        for part in getattr(content, "parts", []) or []
        if getattr(part, "text", None) is not None
    )


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _execute_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    spec = TOOL_REGISTRY.get(name)
    if spec is None:
        return {"error": f"Herramienta desconocida: {name}"}
    try:
        result = spec.function(**args)
        if isinstance(result, dict):
            return _json_safe(result)
        return {"results": _json_safe(result)}
    except Exception as error:
        return {"error": str(error)}


def _append_tool_result(messages: list[Any], call: Any, result: dict[str, Any]) -> None:
    function_response = types.FunctionResponse(
        name=call.name,
        response=result,
        id=getattr(call, "id", None),
    )
    messages.append(
        types.Content(
            role="tool",
            parts=[types.Part(function_response=function_response)],
        )
    )


def _interaction_type(tool_names: list[str]) -> Literal["datos", "politica", "ambos", "rechazo"]:
    has_data = any(name in DATA_TOOL_NAMES for name in tool_names)
    has_policy = "search_policies" in tool_names
    if has_data and has_policy:
        return "ambos"
    if has_data:
        return "datos"
    if has_policy:
        return "politica"
    return "rechazo"


_NUMBER_RE = re.compile(r"(?<!\d)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_SECTION_RE = re.compile(r"§\s*\d+(?:\.\d+)*")


def _numbers(text: str) -> set[str]:
    text = _DATE_RE.sub(" ", text)
    text = _SECTION_RE.sub(" ", text)
    numbers: set[str] = set()
    for match in _NUMBER_RE.finditer(text):
        value = match.group()
        prefix = text[: match.start()]
        if prefix.endswith(("-", "+")):
            preceding = prefix[-2] if len(prefix) > 1 else ""
            if not preceding or not preceding.isalnum():
                value = prefix[-1] + value
        try:
            numbers.add(str(Decimal(value.replace(",", "")).normalize()))
        except Exception:
            numbers.add(value.replace(",", ""))
    return numbers


def _grounded(answer_text: str, question: str, tool_results: list[dict[str, Any]]) -> bool:
    source = question + " " + json.dumps(tool_results, ensure_ascii=False, default=str)
    return _numbers(answer_text).issubset(_numbers(source))


def _log_interaction(
    interaction_type: str,
    tool_names: list[str],
    tool_calls: int,
    token_totals: dict[str, int],
) -> None:
    INTERACTIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "type": interaction_type,
        "tool_names": list(dict.fromkeys(tool_names)),
        "tool_calls": tool_calls,
        "input_tokens": token_totals["input_tokens"],
        "output_tokens": token_totals["output_tokens"],
        "thinking_tokens": token_totals["thinking_tokens"],
    }
    with INTERACTIONS_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _log_grounding_failure() -> None:
    ROUTER_ERRORS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "numeric_grounding_failed",
    }
    with ROUTER_ERRORS_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


def _refusal() -> str:
    return "No tengo esa información. Para responderla necesitaría datos verificables en las fuentes disponibles."


def answer(question: str, history: list[Any] | None = None) -> str:
    """Answer one question using up to five rounds of deterministic tools."""
    messages = list(history or [])[-6:]
    messages.append(types.Content(role="user", parts=[types.Part(text=question)]))
    used_tools: list[str] = []
    tool_results: list[dict[str, Any]] = []
    usage_records: list[dict[str, object]] = []
    token_totals = {"input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    number_of_tool_calls = 0
    tool_rounds = 0
    final_answer = _refusal()

    try:
        while True:
            response, usage = llm.generate(
                messages,
                tools=FUNCTION_TOOLS if tool_rounds < MAX_TOOL_ROUNDS else None,
                system_instruction=SYSTEM_PROMPT,
                interaction_type="unknown",
            )
            usage_records.append(usage)
            for key in token_totals:
                token_totals[key] += int(usage.get(key, 0))

            response_content = _content(response)
            if response_content is not None:
                messages.append(response_content)
            calls = _function_calls(response)
            if not calls:
                final_answer = _response_text(response)
                break
            if tool_rounds >= MAX_TOOL_ROUNDS:
                final_answer = _refusal()
                break

            tool_rounds += 1
            for call in calls:
                tool_name = str(call.name)
                used_tools.append(tool_name)
                number_of_tool_calls += 1
                raw_args = getattr(call, "args", None) or {}
                args = dict(raw_args)
                result = _execute_tool(tool_name, args)
                tool_results.append(result)
                _append_tool_result(messages, call, result)

        if not used_tools:
            final_answer = _refusal()
        elif not _grounded(final_answer, question, tool_results):
            correction = types.Content(
                role="user",
                parts=[
                    types.Part(
                        text="Corrige tu respuesta: elimina cualquier cifra que no aparezca literalmente en la pregunta o en los resultados de herramientas. Responde brevemente y no agregues cifras nuevas."
                    )
                ],
            )
            messages.append(correction)
            response, usage = llm.generate(
                messages,
                system_instruction=SYSTEM_PROMPT,
                interaction_type="unknown",
            )
            usage_records.append(usage)
            for key in token_totals:
                token_totals[key] += int(usage.get(key, 0))
            response_content = _content(response)
            if response_content is not None:
                messages.append(response_content)
            retry_answer = _response_text(response)
            if _function_calls(response) or not _grounded(retry_answer, question, tool_results):
                _log_grounding_failure()
                final_answer = _refusal()
            else:
                final_answer = retry_answer
    #except Exception:
    #    final_answer = "No pude consultar la información disponible. Verifica la configuración e inténtalo de nuevo."
    except Exception:
        import traceback; traceback.print_exc()
        final_answer = "No pude consultar la información disponible. Verifica la configuración e inténtalo de nuevo."
    except Exception as error:
        if os.getenv("LLM_DEBUG") == "1":
            print(f"[router error] {type(error).__name__}: {error}")
        if llm._is_retryable(error):
            final_answer = "El servicio del modelo está saturado, intenta de nuevo en un momento."
        else:
            final_answer = "No pude consultar la información disponible. Verifica la configuración e inténtalo de nuevo."
    finally:
        interaction_type = _interaction_type(used_tools)
        llm.finalize_usage_records(usage_records, interaction_type)
        _log_interaction(interaction_type, used_tools, number_of_tool_calls, token_totals)

    return final_answer