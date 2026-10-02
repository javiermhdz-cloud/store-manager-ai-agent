"""Chat de Streamlit para el asistente de tiendas.

Ejecutar desde la raíz del repositorio:
    streamlit run src/app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from google.genai import types

from src.router import answer

INTERACTIONS_LOG = ROOT / "logs" / "interactions.jsonl"
MAX_HISTORY = 6

EXAMPLES = [
    "¿Cuánto vendió el SKU-1041 en la tienda T01 el 1 de junio de 2026?",
    "¿Con cuántos días de anticipación se retiran los lácteos del anaquel?",
    "¿Qué productos están por debajo de su punto de reorden en la tienda T02?",
    "¿Cuántos tickets hay sobre fugas de agua?",
    "¿Cuánto vamos a vender en octubre?",
]


def _log_lines() -> list[str]:
    if not INTERACTIONS_LOG.exists():
        return []
    with INTERACTIONS_LOG.open(encoding="utf-8") as log_file:
        return [line for line in log_file if line.strip()]


def _new_interaction(lines_before: int) -> dict | None:
    """Return the log record written by the last question, if any."""
    lines = _log_lines()
    if len(lines) <= lines_before:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def _history() -> list[types.Content]:
    """Previous turns in the format the router expects."""
    turns = []
    for message in st.session_state.messages[-MAX_HISTORY:]:
        role = "user" if message["role"] == "user" else "model"
        turns.append(types.Content(role=role, parts=[types.Part(text=message["content"])]))
    return turns


def _show_sources(meta: dict | None) -> None:
    if not meta:
        return
    tools = meta.get("tools") or meta.get("tool_names") or []
    with st.expander("¿De dónde salió esta respuesta?"):
        st.write(f"Tipo de consulta: **{meta.get('type', 'desconocido')}**")
        st.write("Herramientas usadas: " + (", ".join(tools) if tools else "ninguna"))
        st.caption(
            "Las cifras salen de las herramientas de datos, no del modelo. "
            "Las políticas citan documento, sección y versión."
        )


st.set_page_config(page_title="Asistente de tiendas", page_icon="🛒")
st.title("Asistente de tiendas")
st.caption("Pregunta sobre ventas, inventario, tickets y políticas de tus tiendas.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Preguntas de ejemplo")
    clicked = None
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            clicked = example
    if st.button("Nueva conversación"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _show_sources(message.get("meta"))

prompt = st.chat_input("Escribe tu pregunta...") or clicked

if prompt:
    history = _history()
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Consultando los datos y las políticas..."):
            lines_before = len(_log_lines())
            try:
                reply = answer(prompt, history=history)
            except Exception:
                reply = "Ocurrió un error inesperado. Intenta de nuevo en un momento."
            meta = _new_interaction(lines_before)
        st.markdown(reply)
        _show_sources(meta)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "assistant", "content": reply, "meta": meta})