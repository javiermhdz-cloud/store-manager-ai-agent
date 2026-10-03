# Asistente de tiendas

Chat en español para una gerente de tienda que administra cuatro sucursales. Responde preguntas sobre **ventas, inventario, tickets de la mesa de servicio y políticas internas**, con tres reglas:

1. **Las cifras salen de los datos, no del modelo.** El modelo solo decide qué consulta correr; el código calcula el resultado sobre los CSV. Si una respuesta contiene un número que no aparece en la pregunta ni en el resultado de una herramienta, el sistema la rechaza.
2. **Las políticas citan su fuente**: documento, versión y sección, por ejemplo *(Política de mermas y caducidad v3.2, §5.2)*.
3. **Cuando no sabe, lo dice**: responde "No tengo esa información" y explica qué falta (fechas fuera del periodo, tiendas inexistentes, temas que no están en los documentos).

Todos los datos y documentos son sintéticos (ver `BRIEF.md`).

## Correrlo en menos de 10 minutos

Requisitos: Python 3.11 o superior y una API key gratuita de [Google AI Studio](https://aistudio.google.com) (botón *Get API key*).

```bash
git clone https://github.com/javiermhdz-cloud/store-manager-ai-agent.git
cd store-manager-ai-agent
pip install -r requirements.txt

export GEMINI_API_KEY="tu-clave"
export LLM_MODEL=gemma-4-31b-it
export LLM_FALLBACK_MODEL=gemini-3.5-flash

streamlit run src/app.py
```

Se abre un chat con preguntas de ejemplo en la barra lateral. Debajo de cada respuesta hay un panel "¿De dónde salió esta respuesta?" con el tipo de consulta y las herramientas usadas.

En GitHub Codespaces, guarda la clave como secreto `GEMINI_API_KEY` (Settings → Codespaces → Secrets) y reinicia el codespace; no hace falta el `export` de la clave.

### Variables de entorno

| Variable | Para qué sirve | Valor por defecto |
|---|---|---|
| `GEMINI_API_KEY` | Clave de la API (obligatoria) | — |
| `LLM_MODEL` | Modelo principal | `gemini-3.5-flash` |
| `LLM_FALLBACK_MODEL` | Modelo de respaldo si el principal falla por saturación (503/500) | sin respaldo |
| `LLM_THINKING_LEVEL` | Nivel de razonamiento de los modelos Gemini (no aplica a Gemma) | `low` |
| `LLM_MIN_INTERVAL_SECONDS` | Pausa mínima, en segundos, entre llamadas al modelo | `0` (sin pausa) |
| `LLM_DEBUG` | Con `1`, imprime el tipo de error del router (nunca la pregunta ni la clave) | apagado |

**Sobre el modelo.** Se probó con el plan gratuito. `gemini-3.5-flash` es más rápido pero, al momento de escribir esto, su cuota gratuita era de 20 solicitudes por día, que no alcanza para una evaluación; por eso el modelo principal recomendado es `gemma-4-31b-it` (más lento, con más cuota) y Gemini queda de respaldo. Cambiar de modelo es cambiar una variable; todas las llamadas pasan por `src/llm.py`.

## Cómo funciona

```
Pregunta ──► src/router.py ──► el modelo elige herramientas (function calling)
                │                      │
                │            ┌─────────┴───────────┐
                │      src/data_tools.py      src/policies.py
                │      ventas, inventario,    PDFs por sección,
                │      tickets (pandas)       búsqueda BM25
                ▼
        verificación numérica ──► respuesta con fuente, o "No tengo esa información"
```

- **`src/data_tools.py`**: funciones deterministas sobre los CSV (`sales_summary`, `rank_skus`, `sku_info`, `stock_below_reorder`, `lots_expiring_soon`, `ticket_counts`, `list_tickets`, `search_tickets`). Cada resultado incluye los filtros aplicados y, en las listas, un `total` para que el modelo nunca tenga que contar.
- **`src/policies.py`**: extrae los 4 PDFs, los divide por sección y busca con BM25 (sin acentos, con singulares). Cada fragmento conserva documento, sección, página y versión. Se ignoran las secciones "Control de cambios".
- **`src/router.py`**: ciclo de herramientas (máximo 5 rondas), reglas en el prompt del sistema, historial de los últimos 6 mensajes para preguntas de seguimiento ("¿y en la tienda T02?"), y verificación numérica de la respuesta final.
- **`src/llm.py`**: único punto de contacto con el modelo. Reintentos con espera creciente en errores 429/500/503, modelo de respaldo y registro de tokens.
- **`src/app.py`**: chat en Streamlit.

### Datos y fechas

Ventas: 1 jun – 31 ago de 2026. Inventario: una sola foto al 1 sep de 2026. Las fechas límite viven en `src/config.py`. Las devoluciones son filas negativas en ventas y se incluyen por defecto; cada respuesta de ventas dice si las incluye.

## Pruebas y evaluación

```bash
python -m pytest -q
```

Las pruebas usan un modelo simulado: no llaman a la API ni gastan cuota.

```bash
python eval/run_eval.py                # corre lo que falta
python eval/run_eval.py --only datos   # una categoría
python eval/run_eval.py --tag final    # guarda resultados aparte
python eval/run_eval.py --regrade      # recalifica respuestas guardadas, sin llamar al modelo
```

`eval/questions.json` tiene 36 preguntas con respuesta esperada **calculada desde los CSV y los PDF**, no por un modelo: 19 de datos, 9 de políticas, 6 que deben rechazarse y 2 de seguimiento. El script califica cada respuesta con reglas simples (cifras esperadas, frases, cita con documento y `§`), cuenta aparte los errores de servicio, y escribe `eval/results*.jsonl` y `eval/report*.md` con exactitud por categoría, tokens y segundos por tipo de pregunta. Es reanudable: si se agota la cuota, se detiene y la siguiente corrida continúa donde se quedó.

## Registros

`logs/` (no se sube al repositorio) guarda el consumo de tokens por llamada y un renglón por pregunta con su tipo, las herramientas usadas y los tokens. No se guarda el texto de las preguntas ni de las respuestas.

## Límites conocidos

- **No hay herramienta de devoluciones**: "¿cuántas unidades se devolvieron?" se rechaza con honestidad en lugar de calcularse.
- **La verificación numérica solo revisa dígitos**: una cifra escrita con letras ("tres") no se puede verificar.
- **Velocidad**: con el modelo gratuito, una pregunta de datos tarda del orden de uno o dos minutos. Es el principal costo de usar el plan gratuito.
- **Contradicción en los documentos**: el procedimiento de devoluciones dice en §4.1 que un defecto de fabricación fuera de plazo lo autoriza el Gerente de Servicio al Cliente y en §9.1 que lo autoriza el Gerente de Tienda o el Subgerente en turno. El asistente cita ambas, pero no avisa que se contradicen.
- Una pregunta de seguimiento sin contexto previo se rechaza en lugar de pedir aclaración.

## Estructura

```
BRIEF.md, plantilla_decisiones.md
datos/            CSV y diccionario de datos
politicas/        4 PDF de políticas
src/              config, data_tools, policies, llm, router, app
tests/            pruebas (modelo simulado)
eval/             preguntas, script de evaluación, resultados y reportes
scripts/          smoke_llm.py (prueba rápida de la clave y el modelo)
.github/          instrucciones para Copilot
```