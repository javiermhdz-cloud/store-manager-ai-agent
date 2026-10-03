# Documento de decisiones

Fecha: 3 de octubre del 2026
Repositorio: https://github.com/javiermhdz-cloud/store-manager-ai-agent
Proveedor y modelos usados: Google AI Studio (API de Gemini, plan gratuito): `gemma-4-31b-it` como modelo principal y `gemini-3.5-flash` como respaldo. Desarrollo asistido con GitHub Copilot (en Codespaces) y Claude.

Llena cada sección en 3 a 5 líneas. No hace falta que sea prosa pulida; nos interesa el razonamiento.

## 1. Qué construí y por qué en ese orden

- Primero las herramientas de datos (`data_tools.py`) con pruebas: la regla más riesgosa del problema es que ninguna cifra se invente, y la resolví con código determinista antes de tocar un modelo.
- Segundo, el índice de políticas (`policies.py`): los PDFs divididos por sección, búsqueda BM25, y cada fragmento con documento, versión y sección para poder citar.
- Tercero, el envoltorio del modelo (`llm.py`: reintentos, modelo de respaldo, registro de tokens) y después el router: el modelo elige herramientas y una verificación numérica rechaza cualquier cifra que no venga de una herramienta.
- Al final, la interfaz (Streamlit) y el set de evaluación. Dejé el modelo para el final para poder probar cada pieza sin gastar cuota.

## 2. Qué decidí no hacer

- No usé text-to-SQL libre ni embeddings: con 4 PDFs cortos alcanza BM25 por sección, y las consultas de datos son funciones acotadas, más fáciles de probar y de explicar.
- No hice una herramienta de devoluciones: "¿cuántas unidades se devolvieron?" se rechaza con honestidad (pregunta D08) en lugar de calcularse mal.
- No resolví la contradicción entre §4.1 y §9.1 del procedimiento de devoluciones (autoriza el Gerente de Servicio al Cliente vs. el Gerente de Tienda o el Subgerente); el asistente cita ambas.
- Los registros no guardan el texto de preguntas ni respuestas. Tampoco hice autenticación, base de datos ni despliegue: no hacían falta para el ejercicio.

## 3. Cómo sé que funciona

- Un set de 36 preguntas (`eval/`) con respuestas esperadas calculadas desde los CSV y los PDF, no por un modelo: 19 de datos, 9 de políticas, 6 que deben rechazarse y 2 de seguimiento. Un script las califica y cuenta aparte los errores de servicio.
- Primera corrida completa: 29 de 36 (81%). Los fallos eran conteos: las herramientas devolvían listas y el modelo no podía contarlas sin que la verificación numérica rechazara el número. Agregué `total` a las herramientas y quité una regla del prompt que metía ruido.
- Corrida final con `gemma-4-31b-it`: **34 de 36 (94%)**: datos 17/19, políticas 9/9 (10 de 10 citas con documento y sección correctos), rechazos 6/6, seguimiento 2/2.
- Los dos fallos: D08 (no hay herramienta de devoluciones) e I03: el modelo leyó "del 1 al 8" como 8 días, pidió la consulta hasta el 9 de septiembre, vio que el total no coincidía y se negó. Es un problema de la interfaz de la herramienta, no de los datos.
- Además hay pruebas unitarias con un modelo simulado (no usan cuota). Límites: una sola corrida (el modelo no es determinista; I03 había pasado en una corrida parcial) y el calificador solo busca las cifras esperadas en la respuesta.

## 4. Costo por interacción y cómo lo medí

- Cada llamada al modelo registra tokens de entrada, salida y razonamiento; el router escribe una línea por pregunta (tipo, herramientas, tokens, sin texto) y el script de evaluación promedia por tipo. El razonamiento se cobra como salida.
- Corrida final (36 preguntas, `gemma-4-31b-it`), promedio por pregunta: 5,077 tokens de entrada, 83 de salida y 405 de razonamiento, en 103 s. Datos: 5,818 / 97 / 564 y 112 s. Políticas: 5,274 / 79 / 187 y 118 s. Rechazos: 1,681 / 41 / 244 y 37 s.
- Costo real: $0 (plan gratuito). Costo hipotético con tarifas de lista de Gemini 3.5 Flash ($1.50 entrada / $9.00 salida por millón de tokens): **$0.012 por pregunta** (datos $0.0147, políticas $0.0103, rechazos $0.0051), unos $12 por 1,000 preguntas. Con Gemini 3.5 Flash-Lite ($0.30 / $2.50): $0.0027 por pregunta, unos $2.70 por 1,000.
- Lo que más pesa es la entrada (~5 mil tokens: instrucciones, definiciones de herramientas y resultados). El costo real del plan gratuito es el tiempo: ~100 s por pregunta y, en Gemini, 20 solicitudes por día.
- Las tarifas son las de lista que consulté en octubre de 2026. Las llamadas fallidas (sin tokens) no entran en los promedios.

## 5. Qué haría con dos semanas más

- Corregir I03 con un parámetro de fecha final en `lots_expiring_soon` (que el modelo no tenga que convertir fechas en "días desde el corte") y agregar una herramienta de devoluciones para D08.
- Bajar el tiempo de respuesta: prompt y definiciones de herramientas más cortos, caché de prompt, y probar `gemma-4-26b-a4b-it` y Flash-Lite contra el mismo set.
- Que el asistente avise cuando dos documentos se contradicen (§4.1 vs §9.1) y pida aclaración ante un seguimiento sin contexto.
- Repetir cada pregunta 3 veces para medir la variación, ampliar el set y calificar con reglas más estrictas (por ejemplo, detectar cifras de más). La verificación numérica también debería cubrir cifras escritas con letras.
