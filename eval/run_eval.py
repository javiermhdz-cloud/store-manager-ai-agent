"""Evaluación del asistente de tiendas.

Corre las preguntas de eval/questions.json contra el router, califica cada
respuesta con reglas simples y escribe eval/results.jsonl y eval/report.md.

Uso (desde la raíz del repositorio):
    python eval/run_eval.py                      # corre lo que falta
    python eval/run_eval.py --only politica      # una categoría
    python eval/run_eval.py --ids D01,P04        # preguntas específicas
    python eval/run_eval.py --redo               # vuelve a correr todo
    python eval/run_eval.py --price-in 0.30 --price-out 2.50   # USD por 1M tokens

Es reanudable: si el servicio se satura o se agota la cuota, se detiene y
la siguiente corrida continúa donde se quedó. Las preguntas que fallaron por
el servicio se reintentan solas.
"""

import argparse
import json
import sys
import time
import unicodedata
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
RESULTS_PATH = ROOT / "eval" / "results.jsonl"
REPORT_PATH = ROOT / "eval" / "report.md"
INTERACTIONS_LOG = ROOT / "logs" / "interactions.jsonl"

REFUSAL_PREFIX = "no tengo esa informacion"
SERVICE_PREFIXES = (
    "el servicio del modelo esta saturado",
    "no pude consultar la informacion",
)
NUMBER_RE = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


WORD_UNITS = {
    "cero": 0, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7,
    "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "veintiun": 21, "veintiuno": 21, "veintidos": 22,
    "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
    "veintisiete": 27, "veintiocho": 28, "veintinueve": 29, "cien": 100,
}
WORD_TENS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60,
             "setenta": 70, "ochenta": 80, "noventa": 90}
WORD_COMPOUND = {"un": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5,
                 "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}


def word_numbers(text: str) -> list[float]:
    """Numbers written in Spanish words (0-100): 'tres', 'treinta y cinco'."""
    tokens = re.findall(r"[a-z]+", norm(text))
    values: list[float] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in WORD_TENS:
            value = WORD_TENS[token]
            if i + 2 < len(tokens) and tokens[i + 1] == "y" and tokens[i + 2] in WORD_COMPOUND:
                value += WORD_COMPOUND[tokens[i + 2]]
                i += 2
            values.append(float(value))
        elif token in WORD_UNITS:
            values.append(float(WORD_UNITS[token]))
        i += 1
    return values


def numbers_in(text: str) -> list[float]:
    values = []
    for match in NUMBER_RE.finditer(text):
        try:
            values.append(float(match.group().replace(",", "")))
        except ValueError:
            pass
    return values + word_numbers(text)


def grade(question: dict, answer: str) -> tuple[str, list[str], bool | None]:
    """Return (status, reasons, cited). Status: PASS, FAIL or SERVICE_ERROR."""
    text = norm(answer).strip()
    if text.startswith(SERVICE_PREFIXES):
        return "SERVICE_ERROR", ["servicio no disponible"], None
    refused = text.startswith(REFUSAL_PREFIX)

    if question.get("expect_refusal"):
        if refused:
            return "PASS", [], None
        return "FAIL", ["debía rechazar y respondió"], None

    reasons: list[str] = []
    if refused:
        reasons.append("rechazo indebido")
    else:
        found = numbers_in(answer)
        for value, tolerance in question.get("numbers", []):
            if not any(abs(n - value) <= tolerance for n in found):
                reasons.append(f"falta la cifra {value}")
        for group in question.get("contains", []):
            if not any(norm(option) in text for option in group):
                reasons.append("falta: " + " / ".join(group))

    cited = None
    if question.get("cite_docs_any"):
        cited = "§" in answer and any(norm(doc) in text for doc in question["cite_docs_any"])
        if not cited and not refused:
            reasons.append("cita incompleta (documento y §)")
    return ("PASS" if not reasons else "FAIL"), reasons, cited


def _log_lines() -> list[str]:
    if not INTERACTIONS_LOG.exists():
        return []
    with INTERACTIONS_LOG.open(encoding="utf-8") as log_file:
        return [line for line in log_file if line.strip()]


def _new_interaction(lines_before: int) -> dict:
    lines = _log_lines()
    if len(lines) <= lines_before:
        return {}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return {}


def _build_history(items: list[dict]):
    from google.genai import types

    return [
        types.Content(
            role="user" if item["role"] == "user" else "model",
            parts=[types.Part(text=item["text"])],
        )
        for item in items
    ]


def run_one(question: dict, answer_fn) -> dict:
    import os

    history = _build_history(question["history"]) if question.get("history") else None
    lines_before = len(_log_lines())
    started = time.time()
    try:
        answer = answer_fn(question["question"], history=history)
    except Exception as error:  # noqa: BLE001 - the eval must never crash midway
        answer = f"No pude consultar la información disponible. ({type(error).__name__})"
    seconds = round(time.time() - started, 1)
    meta = _new_interaction(lines_before)
    status, reasons, cited = grade(question, answer)
    return {
        "id": question["id"],
        "category": question["category"],
        "status": status,
        "reasons": reasons,
        "cited": cited,
        "answer": answer,
        "seconds": seconds,
        "type": meta.get("type"),
        "tools": meta.get("tool_names", []),
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "thinking_tokens": meta.get("thinking_tokens", 0),
        "model": os.getenv("LLM_MODEL", ""),
    }


def load_results(path: Path = RESULTS_PATH) -> dict[str, dict]:
    results: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as results_file:
            for line in results_file:
                if line.strip():
                    record = json.loads(line)
                    results[record["id"]] = record  # the latest record wins
    return results


def _pct(part: int, total: int) -> str:
    return f"{100 * part / total:.0f}%" if total else "-"


def summarize(questions: list[dict], results: dict[str, dict], price_in, price_out) -> str:
    by_id = {q["id"]: q for q in questions}
    done = [results[q["id"]] for q in questions if q["id"] in results]
    lines = ["# Resultados de la evaluación", ""]
    lines.append(f"Preguntas con resultado: {len(done)} de {len(questions)}.")
    models = sorted({r["model"] for r in done if r.get("model")})
    if models:
        lines.append(f"Modelo(s): {', '.join(models)}.")
    lines += ["", "## Exactitud por categoría", "",
              "| Categoría | Preguntas | Correctas | Incorrectas | Error de servicio | Exactitud (sin errores de servicio) |",
              "|---|---|---|---|---|---|"]
    categories = defaultdict(list)
    for record in done:
        categories[record["category"]].append(record)
    for category, records in categories.items():
        passed = sum(r["status"] == "PASS" for r in records)
        failed = sum(r["status"] == "FAIL" for r in records)
        service = sum(r["status"] == "SERVICE_ERROR" for r in records)
        lines.append(f"| {category} | {len(records)} | {passed} | {failed} | {service} | {_pct(passed, passed + failed)} |")
    passed = sum(r["status"] == "PASS" for r in done)
    failed = sum(r["status"] == "FAIL" for r in done)
    service = sum(r["status"] == "SERVICE_ERROR" for r in done)
    lines.append(f"| **Total** | {len(done)} | {passed} | {failed} | {service} | {_pct(passed, passed + failed)} |")

    policy = [r for r in done if r["category"] in ("politica",) or r.get("cited") is not None]
    graded_policy = [r for r in policy if r["status"] != "SERVICE_ERROR"]
    if graded_policy:
        cited = sum(1 for r in graded_policy if r.get("cited"))
        lines += ["", f"Citas correctas (documento y sección) en preguntas de política: {cited} de {len(graded_policy)}."]
    refusal_questions = [r for r in done if by_id[r["id"]].get("expect_refusal") and r["status"] != "SERVICE_ERROR"]
    if refusal_questions:
        ok = sum(r["status"] == "PASS" for r in refusal_questions)
        lines.append(f"Rechazos correctos en preguntas sin respuesta posible: {ok} de {len(refusal_questions)}.")
    false_refusals = [r["id"] for r in done if "rechazo indebido" in r["reasons"]]
    lines.append(f"Rechazos indebidos (se negó a responder algo que sí podía): {len(false_refusals)}"
                 + (f" ({', '.join(false_refusals)})." if false_refusals else "."))

    usable = [r for r in done if r["status"] != "SERVICE_ERROR" and (r["input_tokens"] or r["output_tokens"])]
    if usable:
        lines += ["", "## Tokens y tiempo por pregunta", "",
                  "| Tipo | Preguntas | Entrada (prom.) | Salida (prom.) | Razonamiento (prom.) | Segundos (prom.)"
                  + (" | Costo USD (prom.) |" if price_in is not None and price_out is not None else " |"),
                  "|---|---|---|---|---|---|" + ("---|" if price_in is not None and price_out is not None else "")]
        by_type = defaultdict(list)
        for record in usable:
            by_type[record["type"] or "sin tipo"].append(record)
        groups = list(by_type.items()) + [("todas", usable)]
        for kind, records in groups:
            n = len(records)
            avg = lambda key: sum(r[key] for r in records) / n
            row = f"| {kind} | {n} | {avg('input_tokens'):.0f} | {avg('output_tokens'):.0f} | {avg('thinking_tokens'):.0f} | {avg('seconds'):.1f} |"
            if price_in is not None and price_out is not None:
                cost = sum((r["input_tokens"] * price_in + (r["output_tokens"] + r["thinking_tokens"]) * price_out) / 1e6 for r in records) / n
                row += f" {cost:.5f} |"
            lines.append(row)
        lines.append("")
        lines.append("Los tokens de razonamiento se cobran como salida. Las preguntas con error de servicio no se promedian.")

    failures = [r for r in done if r["status"] == "FAIL"]
    if failures:
        lines += ["", "## Preguntas incorrectas", ""]
        for record in failures:
            question = by_id[record["id"]]
            tag = " (difícil)" if question.get("hard") else ""
            lines.append(f"- **{record['id']}**{tag}: {question['question']}  \n  Motivo: {'; '.join(record['reasons'])}")
    pending = [q["id"] for q in questions if q["id"] not in results or results[q["id"]]["status"] == "SERVICE_ERROR"]
    if pending:
        lines += ["", f"Pendientes o con error de servicio: {', '.join(pending)}. Vuelve a correr el script para completarlas."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa el asistente de tiendas.")
    parser.add_argument("--only", help="categorías separadas por coma: datos, politica, rechazo, seguimiento")
    parser.add_argument("--ids", help="ids separados por coma, por ejemplo D01,P04")
    parser.add_argument("--limit", type=int, help="máximo de preguntas a correr en esta corrida")
    parser.add_argument("--redo", action="store_true", help="vuelve a correr preguntas ya calificadas")
    parser.add_argument("--delay", type=float, default=8.0, help="segundos de pausa entre preguntas")
    parser.add_argument("--max-service-errors", type=int, default=3, help="se detiene tras N errores de servicio seguidos")
    parser.add_argument("--tag", help="guarda resultados en eval/results_<tag>.jsonl y eval/report_<tag>.md (para comparar modelos)")
    parser.add_argument("--regrade", action="store_true", help="vuelve a calificar las respuestas guardadas sin llamar al modelo")
    parser.add_argument("--price-in", type=float, help="USD por millón de tokens de entrada")
    parser.add_argument("--price-out", type=float, help="USD por millón de tokens de salida")
    args = parser.parse_args()

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.only:
        wanted = {c.strip() for c in args.only.split(",")}
        questions = [q for q in questions if q["category"] in wanted]
    if args.ids:
        wanted_ids = {i.strip() for i in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted_ids]

    results_path, report_path = RESULTS_PATH, REPORT_PATH
    if args.tag:
        results_path = ROOT / "eval" / f"results_{args.tag}.jsonl"
        report_path = ROOT / "eval" / f"report_{args.tag}.md"

    results = load_results(results_path)

    if args.regrade:
        changed = 0
        for question in questions:
            record = results.get(question["id"])
            if not record or record["status"] == "SERVICE_ERROR":
                continue
            status, reasons, cited = grade(question, record["answer"])
            if (status, reasons, cited) != (record["status"], record["reasons"], record.get("cited")):
                changed += 1
                record = dict(record, status=status, reasons=reasons, cited=cited)
                results[question["id"]] = record
                with results_path.open("a", encoding="utf-8") as results_file:
                    results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Recalificadas: {changed} respuestas cambiaron de resultado.")
        report = summarize(questions, results, args.price_in, args.price_out)
        report_path.write_text(report, encoding="utf-8")
        print("\n" + report)
        return

    todo = [q for q in questions
            if args.redo or q["id"] not in results or results[q["id"]]["status"] == "SERVICE_ERROR"]
    if args.limit:
        todo = todo[: args.limit]

    if todo:
        from src.router import answer

        consecutive_errors = 0
        for number, question in enumerate(todo, 1):
            print(f"[{number}/{len(todo)}] {question['id']}: {question['question']}", flush=True)
            record = run_one(question, answer)
            with results_path.open("a", encoding="utf-8") as results_file:
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            results[record["id"]] = record
            detail = f" ({'; '.join(record['reasons'])})" if record["reasons"] else ""
            print(f"    -> {record['status']}{detail} [{record['seconds']} s]", flush=True)
            consecutive_errors = consecutive_errors + 1 if record["status"] == "SERVICE_ERROR" else 0
            if consecutive_errors >= args.max_service_errors:
                print("\nVarios errores de servicio seguidos: el modelo está saturado o se agotó la cuota diaria.")
                print("Se detiene aquí. Vuelve a correr el script más tarde y continuará donde se quedó.")
                break
            if number < len(todo):
                time.sleep(args.delay)
    else:
        print("No hay preguntas pendientes.")

    report = summarize(questions, results, args.price_in, args.price_out)
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()