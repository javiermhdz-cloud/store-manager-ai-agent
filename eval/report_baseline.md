# Resultados de la evaluación

Preguntas con resultado: 36 de 36.
Modelo(s): gemma-4-31b-it.

## Exactitud por categoría

| Categoría | Preguntas | Correctas | Incorrectas | Error de servicio | Exactitud (sin errores de servicio) |
|---|---|---|---|---|---|
| datos | 19 | 12 | 7 | 0 | 63% |
| politica | 9 | 9 | 0 | 0 | 100% |
| rechazo | 6 | 6 | 0 | 0 | 100% |
| seguimiento | 2 | 2 | 0 | 0 | 100% |
| **Total** | 36 | 29 | 7 | 0 | 81% |

Citas correctas (documento y sección) en preguntas de política: 10 de 10.
Rechazos correctos en preguntas sin respuesta posible: 6 de 6.
Rechazos indebidos (se negó a responder algo que sí podía): 1 (D08).

## Tokens y tiempo por pregunta

| Tipo | Preguntas | Entrada (prom.) | Salida (prom.) | Razonamiento (prom.) | Segundos (prom.) |
|---|---|---|---|---|---|
| datos | 20 | 7347 | 134 | 1029 | 167.7 |
| politica | 11 | 4441 | 86 | 198 | 84.5 |
| rechazo | 5 | 1582 | 43 | 240 | 31.6 |
| todas | 36 | 5659 | 107 | 666 | 123.4 |

Los tokens de razonamiento se cobran como salida. Las preguntas con error de servicio no se promedian.

## Preguntas incorrectas

- **D08** (difícil): ¿Cuántas unidades se devolvieron en la tienda T01 en julio de 2026?  
  Motivo: rechazo indebido
- **I01**: ¿Cuántos productos están en o por debajo de su punto de reorden en la tienda T02?  
  Motivo: falta la cifra 3
- **I03**: ¿Cuántos productos de la tienda T01 tienen un lote que caduca entre el 1 y el 8 de septiembre de 2026?  
  Motivo: falta la cifra 17
- **I05**: ¿Cuántos productos de la tienda T04 están por debajo de su punto de reorden?  
  Motivo: falta la cifra 6
- **K01**: ¿Cuántos tickets de prioridad Crítica hay en total?  
  Motivo: falta la cifra 8
- **K03**: ¿Cuántos tickets ha generado la tienda T03?  
  Motivo: falta la cifra 90
- **K06**: ¿Cuántos tickets de prioridad Alta están en estado Abierto?  
  Motivo: falta la cifra 2
