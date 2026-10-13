# Resumen de pruebas — 2026-10-13

Rama actual: `codex/portero-visual-y-reentrenamiento-rl`.
Todo lo siguiente corresponde a simulación MuJoCo con motores B-Human.

## Quién tira contra quién: goles y atajadas

**Nuestro tirador recomendado es v2; nuestro portero de referencia es el
visual v2 `balanced`, con movimientos originales.** En esta tabla, «base»
significa nuestra FSM con motores B-Human. No se ha evaluado el comportamiento
oficial completo del equipo B-Human.

Cada fila muestra un cruce concreto. Los goles son del tirador y las atajadas
son del portero; un tiro fuera/corto o un timeout no cuenta como atajada.

| Quién tira | Contra qué portero | Penales | Goles del tirador | Atajadas del portero | Fuera/cortos | Timeouts | Ensayo |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Nuestro tirador v2** | Base/FSM, motor B-Human | 100 | **93 (93 %)** | 5 | 2 | 0 | A |
| **Nuestro tirador v2** | **Nuestro visual v2 `balanced`** | 100 | **12 (12 %)** | **86** | 2 | 0 | A |
| Tirador base/FSM, motor B-Human | **Nuestro visual v2 `balanced`** | 20 | **2 (10 %)** | **17** | 0 | 1 | B |
| Tirador base/FSM, motor B-Human | Base/FSM, motor B-Human | 200 | **14 (7 %)** | 181 | 4 | 1 | C |
| Nuevo tirador v4, experimental | Base/FSM, motor B-Human | 100 | **93 (93 %)** | 5 | 2 | 0 | A |
| Nuevo tirador v4, experimental | Nuestro visual v2 `balanced` | 100 | **15 (15 %)** | 83 | 2 | 0 | A |

**Lectura directa:** en el ensayo A, nuestro tirador v2 mete 93 de 100 al
portero base y 12 de 100 a nuestro portero. Nuestro portero ataja 86 de esos
100 tiros de v2. Frente al tirador base atajó 17 de 20 en el ensayo B.

Las filas proceden de tres ensayos diferentes; B y C no son una comparación
pareada entre porteros. No se extrapolan sus conteos a 100 ni se mezclan sus
semillas. Los cruces del ensayo A sí usan las mismas semillas y condiciones
iniciales para ambos tiradores dentro de cada rival.

- **A:** prueba independiente del último reentrenamiento: 600 duelos, 100 por
  tirador y rival. [Registros y conteos](validation/rl/striker_preserving_v4/test_report.json).
- **B:** prueba ciega del selector: subconjunto `center`, política `balanced`,
  20 penales. [Reporte](validation/vision/keeper_selector_v1/heldout_report.json).
  El estilo `center` solicita exactamente la misma acción que el tirador
  `scripted`: centro, pie izquierdo, potencia nominal y disparo inmediato.
  Se verificó la igualdad de sus acciones normalizadas; la equivalencia es
  con nuestra política base, no con el equipo oficial B-Human.
- **C:** duelo histórico del 2026-10-10, condición `realista`, cruce
  `scripted|scripted`, 200 penales.
  [Reporte](validation/rl/duelo_tiradores_vs_portero_fsm_20261010.json).

## Ensayo adicional: portero con salto ajustado

Estas dos filas también pertenecen al ensayo A. El movimiento ajustado es
experimental y no sustituye al portero de referencia de la tabla anterior.

| Quién tira | Contra qué portero | Penales | Goles del tirador | Atajadas del portero | Fuera/cortos |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Nuestro tirador v2 | Nuestro visual v2 `balanced`, salto ajustado | 100 | **19 (19 %)** | 79 | 2 |
| Nuevo tirador v4, experimental | Nuestro visual v2 `balanced`, salto ajustado | 100 | **23 (23 %)** | 75 | 2 |

El nuevo tirador anota tres goles más frente al portero de referencia y
cuatro más frente al salto ajustado. Los IC95 pareados de esas diferencias
son −1 a +8 y −2 a +10 puntos porcentuales: incluyen empate.
**v2 sigue recomendado y v4 queda experimental**.

Fuente: [reporte completo de preserving_v4](validation/rl/striker_preserving_v4/README.md).

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
