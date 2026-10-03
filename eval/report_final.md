# Resultados de la evaluación

Preguntas con resultado: 36 de 36.
Modelo(s): gemma-4-31b-it.

## Exactitud por categoría

| Categoría | Preguntas | Correctas | Incorrectas | Error de servicio | Exactitud (sin errores de servicio) |
|---|---|---|---|---|---|
| datos | 19 | 17 | 2 | 0 | 89% |
| politica | 9 | 9 | 0 | 0 | 100% |
| rechazo | 6 | 6 | 0 | 0 | 100% |
| seguimiento | 2 | 2 | 0 | 0 | 100% |
| **Total** | 36 | 34 | 2 | 0 | 94% |

Citas correctas (documento y sección) en preguntas de política: 10 de 10.
Rechazos correctos en preguntas sin respuesta posible: 6 de 6.
Rechazos indebidos (se negó a responder algo que sí podía): 2 (D08, I03).

## Tokens y tiempo por pregunta

| Tipo | Preguntas | Entrada (prom.) | Salida (prom.) | Razonamiento (prom.) | Segundos (prom.) | Costo USD (prom.) |
|---|---|---|---|---|---|---|
| datos | 20 | 5818 | 97 | 564 | 111.9 | 0.01468 |
| politica | 11 | 5274 | 79 | 187 | 117.6 | 0.01031 |
| rechazo | 5 | 1681 | 41 | 244 | 37.1 | 0.00508 |
| todas | 36 | 5077 | 83 | 405 | 103.3 | 0.01201 |

Los tokens de razonamiento se cobran como salida. Las preguntas con error de servicio no se promedian.

## Preguntas incorrectas

- **D08** (difícil): ¿Cuántas unidades se devolvieron en la tienda T01 en julio de 2026?  
  Motivo: rechazo indebido
- **I03**: ¿Cuántos productos de la tienda T01 tienen un lote que caduca entre el 1 y el 8 de septiembre de 2026?  
  Motivo: rechazo indebido
