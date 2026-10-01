# Copilot instructions

## Project

A Spanish-language chat assistant for a store manager who runs four stores. She asks about sales, inventory, service-desk tickets, and internal policies. Full requirements are in `BRIEF.md`. Column definitions are in `datos/diccionario_datos.md`. Read both before changing anything. All data is synthetic.

## Non-negotiable rules

1. **Numbers come from code, never from the LLM.** Every figure shown to the user (sales, units, stock, counts, percentages, dates) must be computed by a Python function over the CSVs in `datos/`. The LLM chooses which function to call and with which arguments, and then explains the result. It must never calculate, estimate, or recall a figure itself.
2. **Policy answers must cite their source.** Answers about policies come only from the PDFs in `politicas/` and must name the document and the section they came from. If no retrieved section supports the answer, do not answer from general knowledge.
3. **Admit what can't be answered.** If the question is outside the data or documents (forecasts, other stores, dates outside 2026-06-01 to 2026-08-31, inventory history, anything not in the files), reply that the information is not available and say what would be needed. Never fill the gap with a plausible guess.
4. **User-facing text is in Spanish** (Mexico). Code, comments, identifiers, and commit messages are in English.
5. **Log token usage on every LLM call**: input tokens, output tokens, model name, and interaction type (`datos`, `politica`, `rechazo`). Costs are reported in `plantilla_decisiones.md`, so this is a deliverable, not an extra.

## Data facts to respect

- Sales cover 2026-06-01 to 2026-08-31. Inventory is a single snapshot at 2026-09-01. Do not present inventory as a time series.
- `ventas.csv` has one row per day, store, SKU, and channel with non-zero sales. Missing days mean zero sales, not missing data.
- Returns are rows with negative `unidades` and `venta_neta_mxn`. State in each answer whether returns are included, and keep one documented default.
- `venta_neta_mxn` is already net of discount. Do not subtract `descuento_mxn` again.
- Only stores with `formato == "Supermercado"` have the `Ecommerce` channel.
- Discontinued SKUs (`estatus == "Descontinuado"`) have no inventory and may have a `sku_sustituto`. Names can change between a SKU and its substitute. Join on `sku`, not on names.
- `fecha_caducidad_lote_proximo` exists only for perishables.
- Tickets have free-text Spanish with deliberate typos. Do not rely on exact keyword matches. Group by the structured fields (`categoria`, `prioridad`, `estado`, `tienda_id`) first.
- Amounts are in MXN. Dates are ISO 8601, local Monterrey time.

## Architecture

- `src/data_tools.py`: pure, tested pandas/DuckDB functions (sales by store/category/SKU/period, top and bottom SKUs, stock below reorder point, lots expiring soon, ticket counts and lists). Each returns a structured result (dict or DataFrame) that includes the filters used.
- `src/policies.py`: extracts the PDFs by section, builds the index, and retrieves chunks. Each chunk keeps `document` and `section` metadata.
- `src/router.py`: classifies each question as `datos`, `politica`, `ambos`, or `rechazo`.
- `src/llm.py`: single wrapper for the model provider. Reads model name and API key from environment variables. Records token usage. No other module calls the provider directly.
- `src/app.py`: Streamlit chat UI, thin, with no business logic.
- `eval/`: `questions.yaml` and `run_eval.py`.
- `tests/`: pytest tests for every function in `data_tools.py`.

Prefer tool calling over free-form text-to-SQL. If SQL generation is ever used, restrict it to read-only queries on known tables and show the query to the user.

## Evaluation

`eval/questions.yaml` has three kinds of questions:

- **Numeric:** expected value computed independently with pandas, with a tolerance.
- **Policy:** expected document and section.
- **Unanswerable:** the correct behavior is a refusal.

`run_eval.py` reports numeric accuracy, citation correctness, refusal rate, and average tokens per interaction type. Do not generate expected answers with the LLM. Compute them from the CSVs and PDFs.

## How to work in this repo

- Plan before coding. For any non-trivial task, propose the approach and the files you will touch, then wait for approval.
- Work in small steps. One module or feature per change.
- Every new function in `data_tools.py` comes with a pytest test whose expected value is checked against a hand-computed number from the CSV.
- Do not add dependencies without saying why. Keep `requirements.txt` minimal and pinned.
- No secrets in code or commits. Use `.env` (gitignored) and provide `.env.example`.
- Keep the README runnable in under 10 minutes: install, set env vars, `streamlit run src/app.py`.
- Explain non-obvious code in short comments. The author must be able to defend every line.

## Style

- Python 3.11+, type hints on public functions, short docstrings.
- Formatting with `ruff` and `black` defaults.
- Prefer clear, boring code over clever abstractions.
- Error messages shown to users are in Spanish, short, and say what to try next.
