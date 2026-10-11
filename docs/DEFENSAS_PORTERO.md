# Catálogo ampliado de defensas y predicción del portero

Rama `codex/mejora-portero`. Se habilitan movimientos originales B-Human en MuJoCo, conservando el predictor publicado `vision_keeper_v2`. Esta ampliación no implica que los nuevos movimientos atajen más: se comparan por separado antes de elegirlos.

## Seis defensas y una preparación

| Nombre del adaptador | Solicitud original | Keyframe | Espejo |
| --- | --- | --- | --- |
| `center` | `squatArmsBackLeft` | `genuflectStand` | No |
| `center_right` | `squatArmsBackRight` | `genuflectStand` | Sí |
| `wide_left` | `squatWideArmsBackLeft` | `genuflectStandDefender` | No |
| `wide_right` | `squatWideArmsBackRight` | `genuflectStandDefender` | Sí |
| `left` | `jumpLeft` | `keeperJumpLeft` | No |
| `right` | `jumpRight` | `keeperJumpLeft` | Sí |
| `prepare` (preparación) | `prepare` | `sitDownKeeper` | No |

Antes había tres respuestas: `center`, `left` y `right`. Ahora el host, la sesión y el panel manual aceptan las seis defensas; la preparación se ofrece aparte. No se modifica ninguna plantilla articular ni fuente de `baseline/bhuman`.

Las guardias centrales y anchas se mantienen mientras se pide bloqueo; los saltos conservan la recuperación automática original. La primera defensa comprometida queda fija: cambiar el subtipo de `dive` durante una fase no garantiza iniciar otra plantilla. La preparación no se activa automáticamente antes del tiro.

El protocolo local sigue en versión 1 y HELLO añade la capacidad `keeper_defenses_v2` y sus nombres. El nuevo benchmark exige esa capacidad para impedir ejecutar una variante con un binario antiguo. El contrato anterior `keeper_action_v1` permanece intacto; `KeeperDefenseAction` publica `keeper_action_v2`, con `defense` y `trigger` y sin amplitud física ficticia.

## Cómo predice v2

1. Detecta el balón desde RGB y lo proyecta al plano del suelo con la calibración y la cámara reconstruida desde encoders, IMU y apoyo de pies propios.
2. Estima velocidad entre capturas y compensa el movimiento propio mediante odometría. Necesita posición y velocidad válidas, visibilidad y edad de medida ≤250 ms.
3. Confirma una velocidad hacia el portero mayor que el ruido: `vx < -(0.12 + 1.5 * sqrt(Var(vx)))`, en m/s, y balón por delante (`x > 0`).
4. Calcula `t = (-0.12 - x) / vx` y `y_cruce = y + vy * t`. Las posiciones son relativas al portero, en metros; `+y` es su izquierda. La línea de defensa está 12 cm detrás del origen local.
5. Confirma dos capturas entrantes distintas y usa la mediana de hasta tres proyecciones. Cruce entre −0.30 y +0.30 m: centro; fuera: lado correspondiente.
6. El centro se activa al confirmar esas dos capturas. Los saltos esperan hasta que queden ≤1.8 s para la llegada. La primera decisión queda fija y el plazo de 28 s desactiva nuevas solicitudes.

Es una extrapolación del **balón ya en movimiento**, no una predicción de intención antes de la patada. v2 y la nueva biblioteca v6 no usan yaw ni posición del rival para elegir la defensa, aunque el pipeline los mida. No modelan curva, altura de tiro ni frenado explícito; la incertidumbre de velocidad filtra falsas llegadas, pero v2 no produce un intervalo de confianza del cruce.

Los valores de sensores son ideales simulados; la cámara usa odometría/actitud estimadas y geometría nominal. El motor de bajo nivel conserva su perfil ideal explícito. La inferencia no hace avanzar el reloj físico de MuJoCo: el ensayo no prueba equivalencia con latencias de un NAO real.

## Biblioteca v6: aislar el efecto del movimiento

`VisionLibraryKeeper` compone el núcleo v2 sin cambiar su trayectoria, umbrales ni momento de compromiso. Recibe `vision_state_v2` y devuelve el nombre de defensa mediante `keeper_action_v2`.

| Modo | Defensa cuando v2 decide centro | Saltos |
| --- | --- | --- |
| `original` | `center`, control de compatibilidad | Iguales a v2 |
| `center_right` | Siempre centro espejado | Iguales a v2 |
| `wide_left` | Siempre guardia ancha izquierda | Iguales a v2 |
| `wide_right` | Siempre guardia ancha derecha | Iguales a v2 |
| `balanced` | `center_right` para cruce negativo, `center` para el resto | Iguales a v2 |

**Disponibilidad de seis defensas no significa que un único modo las elija todas.** `balanced` elige cuatro; las dos guardias anchas son variantes experimentales. Esta entrega v6 no entrenó una política nueva ni movimientos articulares. El experimento posterior de selección entre las seis conserva el núcleo v2 y se documenta aparte en [ENTRENAMIENTO_SELECTOR_PORTERO.md](ENTRENAMIENTO_SELECTOR_PORTERO.md).

En los registros se separan `commit_side` (zona central/izquierda/derecha del núcleo) y `commit_defense` (plantilla realmente solicitada). Una guardia ancha que sustituye el centro sigue siendo una decisión central para comparar el predictor.

El simulador/panel general mantiene su FSM anterior; la biblioteca de cámara se selecciona mediante los comandos siguientes. En modo manual del panel, seleccionar Portero, una defensa y Bloqueo permite revisar la primitiva.

## Ejecutar y observar

```powershell
# Recompilar el host con MSVC 2022 o 2026; verifica los originales.
.venv/Scripts/python.exe tools/build_motion_runtime.py

# Seis defensas, preparación, recuperación y comparativos visuales.
.venv/Scripts/python.exe tools/validate_keeper_defenses.py --output runs/keeper_defenses/motions

# Duelo de cámara/sensores de un penal; añadir --live para abrir MuJoCo.
.venv/Scripts/python.exe tools/watch_keeper_defenses.py --mode balanced --style v2 --seed 72001000000 --output runs/keeper_defenses/watch

# Comparación pequeña de integración: 5 variantes x 8 estilos x 2 casos = 80.
.venv/Scripts/python.exe tools/evaluate_keeper_defenses.py --episodes 2 --seed 72001 --workers 4 --output runs/keeper_defenses/smoke_72001.json

# Evaluación posterior con nuevas semillas; no reajustar con las del test final.
.venv/Scripts/python.exe tools/evaluate_keeper_defenses.py --episodes 50 --seed 73001 --workers 4 --output runs/keeper_defenses/heldout_73001.json
```

El visor opcional coloca las esperas de IPC fuera del mutex del visor. Los fallos de cámara/movimiento cuentan como `sim_error`, y los procesos nativos se recogen incluso al cerrar el visor o ante una excepción. Los tiradores conservan la observación privilegiada con ruido declarada del benchmark anterior; el portero consume píxeles y sensores propios.

## Verificación

Resultados y visualizaciones congelados en [keeper_defenses_v6](validation/vision/keeper_defenses_v6/README.md):

- Suite completa: **221 pruebas pasan** en 85.311 s.
- Ejecución física MuJoCo: **7/7 solicitudes** ejecutan la plantilla original y recuperan una postura estable, con confirmación adicional de apoyo durante 1 s. Una postura inicial y una semilla; no mide alcance máximo ni atajadas.
- Reproducción: **87/87 decisiones** idénticas a la traza; el núcleo y el adaptador publicado v2 conservan lado, predicción y activación. La nueva biblioteca sólo cambia el nombre de la defensa central.
- Duelo de integración: **80 ejecuciones**, cinco variantes sobre **16 casos pareados** (ocho estilos por dos semillas), **cero errores de simulación**.

| Variante | Atajadas / 16 |
| --- | --- |
| Guardia original | 8 |
| Centro espejado | 11 |
| Guardia ancha izquierda | 8 |
| Guardia ancha derecha | 9 |
| Centro elegido según signo del cruce (`balanced`) | 11 |

El centro espejado es un candidato para evaluar con más casos. Dos semillas independientes son insuficientes para interpretar el bootstrap del JSON como prueba de mejora; sus intervalos quedan registrados, pero no fundamentan una conclusión competitiva. Las semillas tampoco ejercitaron el salto derecho en este duelo: su ejecución y recuperación sí se validaron por separado. Los porcentajes de este ensayo no son comparables directamente con los 400 penales del reporte anterior.

Seis movimientos disponibles permiten estudiar la selección de defensa; no se entrenaron trayectorias nuevas. v2 sigue siendo el control publicado y v6 es experimental. Los resultados anteriores corresponden al ensayo de integración v6. La comparación posterior del selector entrenado utiliza nuevas semillas y un reporte independiente, enlazado desde [ENTRENAMIENTO_SELECTOR_PORTERO.md](ENTRENAMIENTO_SELECTOR_PORTERO.md).

Fuentes: `penalty/keeper_defenses.py`, `keeper_library.py`, `rl/defense_benchmark.py`, `native/bhuman_motion/main.cpp`; configuración y mapeos originales de B-Human preservados. Antecedentes: [PORTERO_FUSION.md](PORTERO_FUSION.md).

This product includes software developed by B-Human (http://www.b-human.de).
