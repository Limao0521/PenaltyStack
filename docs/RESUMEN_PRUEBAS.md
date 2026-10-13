# Resumen de pruebas — 2026-10-13

Rama actual: `codex/portero-visual-y-reentrenamiento-rl`.
Todo lo siguiente corresponde a simulación MuJoCo con motores B-Human.

## Tirador: última prueba independiente

**600 penales nuevos**, 100 por tirador y portero, emparejados por semilla
y condiciones iniciales dentro de cada rival. Más goles es mejor para el tirador.

| Portero rival | Tirador v2 | Nuevo tirador v4 | Cambio de goles | IC95 pareado |
| :--- | ---: | ---: | ---: | :--- |
| Visual v2 `balanced`, movimiento original | 12/100 · 12 % | **15/100 · 15 %** | +3 puntos | −1 a +8 puntos |
| Visual v2 `balanced`, salto ajustado | 19/100 · 19 % | **23/100 · 23 %** | +4 puntos | −2 a +10 puntos |
| FSM base propia, motor B-Human | **93/100 · 93 %** | **93/100 · 93 %** | 0 puntos | 0 a 0 puntos |

Cada tirador registró dos tiros fuera/cortos por rival, separados de las
atajadas. El nuevo tirador conserva el rendimiento frente a la FSM y mejora
ligeramente en esta muestra frente a los porteros visuales. Los intervalos
incluyen empate: **v2 sigue recomendado y v4 queda experimental**.

Fuente: [reporte completo de preserving_v4](validation/rl/striker_preserving_v4/README.md)
y [registros de prueba](validation/rl/striker_preserving_v4/test_report.json).

## Comprobaciones del último reentrenamiento

| Comprobación | Resultado |
| :--- | :--- |
| Pruebas automatizadas | **367 de 367 pasan** |
| Paridad física antes de aprender | **4 duelos**, observaciones, acciones y resultados idénticos a v2 |
| Entrenamiento | **96 actualizaciones**, 73 728 transiciones optimizadas, 902 episodios completos |
| Validación | **640 duelos**, cero errores de simulación |
| Prueba independiente | **600 duelos**, cero errores de simulación |
| Checkpoint seleccionado | **Actualización 64**, elegida antes de la prueba; 49 152 transiciones nuevas |
| Fallos registrados del entrenamiento final | **0** |
| Inputs y originales verificados | **121 inputs** y **559 originales B-Human** intactos |

La validación eligió 14/64 goles frente a 13/64 de v2 ante `balanced`;
ambos obtuvieron 60/64 ante la FSM. La corrida inicializada produjo los
mismos resultados que v2 antes de aprender. El error previo de un helper
de guardado ocurrió antes de la comprobación física y se documenta aparte.

Evidencia: [auditoría final](validation/rl/striker_preserving_v4/final_audit.json),
[367 pruebas](validation/rl/striker_preserving_v4/full_tests.log),
[validación](validation/rl/striker_preserving_v4/validation_report.json)
y [corrida completa con checkpoints y RNG](validation/rl/striker_preserving_v4/complete_run.zip).

## Portero: elección entre defensas

Prueba ciega de **160 casos de ocho estilos**, 20 por estilo, con las tres
políticas evaluadas sobre las mismas situaciones. Más atajadas es mejor para el portero.

| Portero | Atajadas | Tasa |
| :--- | ---: | ---: |
| Visual v2 original | 128/160 | 80.000 % |
| Visual v2 `balanced` | **134/160** | **83.750 %** |
| Selector aprendido de seis defensas | 135/160 | 84.375 % |

El selector aprendido gana una atajada respecto a `balanced`: +0.625 puntos,
IC95 de −3.125 a +4.375. No demuestra una mejora adicional fiable;
se conserva como experimental. `Balanced` frente a v2 original gana
3.750 puntos, IC95 de +1.250 a +6.875, en esta muestra.

Fuente: [reporte del selector](validation/vision/keeper_selector_v1/README.md)
y [prueba ciega](validation/vision/keeper_selector_v1/heldout_report.json).

## Portero: ajustes de movimiento

Otra prueba reservada de **160 casos de ocho estilos**, con comparación
pareada entre movimiento original y el perfil ajustado `keeper_motion_v1`.

| Movimiento sobre v2 `balanced` | Atajadas | Tasa |
| :--- | ---: | ---: |
| Original B-Human | 134/160 | 83.75 % |
| Perfil ajustado | 136/160 | 85.00 % |

La ganancia es +1.25 puntos, IC95 de −1.875 a +5.0. El perfil queda
experimental; se conserva el movimiento original como referencia.
El ajuste seleccionado acorta las fases activas del salto; B-Human sigue
ejecutando y equilibrando los movimientos.

Fuente: [reporte de movimiento](validation/vision/keeper_motion_v1/README.md)
y [prueba reservada](validation/vision/keeper_motion_v1/test_report.json).

Las tablas de selector y movimiento usan semillas distintas. Sus tasas se
comparan dentro de cada ensayo. El historial de versiones, incluyendo el
reentrenamiento anterior que perdió rendimiento, permanece en
[RESULTADOS_RL.md](RESULTADOS_RL.md).

## Alcance

El portero visual recibe RGB y sensores propios ideales simulados. El
tirador mantiene información privilegiada de escena con ruido. La FSM base
es del proyecto con motor B-Human; no representa el comportamiento oficial
completo de un equipo B-Human. Estos resultados no validan un NAO físico
ni demuestran convergencia de RL.
