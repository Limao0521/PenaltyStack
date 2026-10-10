# Mejora del portero: leer la postura del tirador

2026-10-10, rama `feat/rl-contracts`. Todas las cifras son de simulación (MuJoCo + motor B-Human original). Ninguna afirma equivalencia con el NAO real. Contexto general del entrenamiento: [RESULTADOS_RL.md](RESULTADOS_RL.md).

## Resumen

- **La postura del tirador delata el lado del tiro.** Desde lo que el portero ya observa (posición del tirador respecto al balón y hacia dónde mira), un clasificador simple acierta el **78 %** de los lados y el **86 %** de los tiros a las esquinas, y ya 2 s antes de la patada acierta el 82 % en las esquinas.
- **El portero `anticipation`** usa esa predicción: cuando el balón se mueve, se lanza al lado predicho si está seguro y, si no, se agacha al centro.
- **Contra el tirador v2 ataja 24 %**, frente a 4 % de la guardia central y 6 % del portero FSM de B-Human. Contra patadas aleatorias empata con la guardia central, y contra el tirador FSM, que patea siempre al centro, rinde un poco menos.
- **El límite es la estirada, no la predicción.** Aun lanzándose al lado correcto, las estiradas de B-Human atajan 46–59 % de los tiros. Un portero que conociera el lado real de antemano ataja 0.62 contra patadas aleatorias.

## 1. Cómo decidía el portero hasta ahora

**El portero entrenado con RL (v4)** es una red neuronal (MLP 2×128, tanh).
- **Qué recibe:** las últimas 6 observaciones (0.6 s), cada una con 19 datos medidos desde el portero:
  - el balón: posición, velocidad, si se ve y hace cuánto se vio;
  - el rival: posición y orientación;
  - los ángulos al centro del arco y a los postes;
  - su propio torso y giróscopo, y el tiempo.
- **Qué devuelve,** cada 0.1 s:
  - el lado: < −1/3 derecha, > 1/3 izquierda, centro en medio;
  - el gatillo: por debajo de 0.5 espera; si no, se compromete una sola vez;
  - una amplitud que B-Human no aplica.
- **Cómo predice:** no hace ninguna predicción física. PPO aprendió la relación "movimiento del balón → lado" y la aprendió mal: elige el lado equivocado el 67 % de las veces y ataja 0.35–0.37.

**Los porteros de referencia** usan reglas explícitas:

| Portero | Cómo elige el lado |
| --- | --- |
| Geométrico | Extrapola el balón con frenado constante (0.6 m/s²) hasta su línea. Faltando ≤ 0.7 s para la llegada, se tira a la izquierda si el cruce previsto está a más de +0.2 m, a la derecha si está a menos de −0.2 m, y al centro en otro caso. |
| FSM de B-Human | Cuando el balón va a más de 0.15 m/s y está a menos de 1.1 m, decide por el ángulo al balón: izquierda si supera 0.18 rad (~10°), derecha si es menor que −0.18, centro si no. |
| Guardia central | Se agacha al centro en cuanto el rival patea. |

**Tipos de lanzamiento.** El portero usa 3 movimientos originales de B-Human (`native/bhuman_motion/main.cpp`):

| Lado | Movimiento | Qué hace |
| --- | --- | --- |
| Izquierda | `jumpLeft` (keyframe `keeperJumpLeft`) | Estirada lateral; termina en el suelo y se levanta solo |
| Derecha | `jumpRight` | Espejo del anterior |
| Centro | `squatArmsBackLeft` | Se agacha en el centro, sin caerse |

B-Human trae 6 movimientos de portero (`MotionRequest::Dive`). No se usan `squatArmsBackRight`, `squatWideArmsBackLeft` ni `squatWideArmsBackRight` (agacharse con las piernas abiertas).

## 2. La idea

Para apuntar fuera del centro, el tirador táctico rodea el balón, así que su posición respecto al balón y su orientación antes de patear dependen del lado al que va a tirar. El portero ya recibía la posición y la orientación del rival, pero el RL nunca aprendió a usarlas: en el mundo sintético donde empieza a entrenar, el rival está quieto y no da pistas.

## 3. Verificación antes de construirlo

599 penales en B-Human, con el rival `random_kicks` pateando a las esquinas (3/4 apuntados a |y| 0.3–0.65 m, 1/4 al centro) y con variación de posiciones (±5 cm de balón, ±5 cm/±0.1 rad de pose inicial). El portero se quedaba quieto mientras se registraba lo que veía.

**Se puede predecir.** Regresión logística sobre la postura del tirador, con validación cruzada de 5 particiones:

| Momento | Acierto (3 lados) | Solo tiros a las esquinas |
| --- | --- | --- |
| Al patear | 77 % | 87 % |
| 0.5 s antes | 78 % | 86 % |
| 1 s antes | 77 % | 85 % |
| 2 s antes | 74 % | 82 % |

Los tiros al centro son los que más confunde. Además, el balón termina en el lado al que apuntó el tirador solo el 73 % de las veces: las patadas de B-Human son imprecisas.

**Anticipar tiene un techo.** Con las mismas patadas, un portero oráculo que conoce el lado real se compromete en la primera decisión después del contacto:

| Lado real | Tiros | Atajadas del oráculo |
| --- | --- | --- |
| Derecha del portero (`jumpRight`) | 189 | 0.59 |
| Centro (agacharse) | 185 | 0.83 |
| Izquierda del portero (`jumpLeft`) | 225 | 0.46 |
| **Total** | 599 | **0.62** |

Sin moverse, el portero ataja 0.28: el balón le pega en el cuerpo. La guardia central tras la patada, con otras 300 patadas del mismo tipo, ataja 0.57.

## 4. Implementación

| Pieza | Qué hace |
| --- | --- |
| `penalty/rl/anticipation.py` | `pose_features` calcula 7 variables desde la observación del portero. `KickSidePredictor` es una regresión logística de 3 clases que se guarda en JSON. `AnticipatingKeeperPolicy` es el portero. |
| `tools/fit_kick_predictor.py` | Junta penales en B-Human, ajusta el predictor con validación cruzada y lo guarda en `models/rl/kick_side_predictor.json`. |
| `models/rl/kick_side_predictor.json` | Predictor ajustado con 566 penales: 78.1 % de acierto, 85.6 % en las esquinas. El archivo guarda sus datos de origen. |
| `penalty/rl/policies/baselines.py`, `training.py` | El portero `anticipation` está en `make_policy` y en los baselines de las etapas del portero. |
| `tests/test_rl_training.py` | Pruebas de las variables, el ajuste, el guardado y la decisión del portero. |

**Variables de la postura** (marco del portero):
- posición del balón (x, y);
- posición del tirador respecto al balón (dx, dy);
- seno y coseno de la orientación del tirador;
- y en que la línea de su mirada, trazada desde el balón, cruza la línea del portero.

**Decisión del portero:**
1. Espera mientras el balón no se mueva (≤ 0.15 m/s en su observación). Así detecta la patada como lo haría el robot y no necesita el filtro `commit_after_shot`.
2. Cuando el balón se mueve, calcula la probabilidad de cada lado. Se lanza a la esquina más probable si esa probabilidad es ≥ 0.75; si no, se agacha al centro.

## 5. Resultados

Penales que el predictor nunca vio, con ruido de percepción. Los reportes están en `docs/validation/rl/anticipacion_20261010/`.

| Portero | vs tirador **v2** (duelo, 200) | vs patadas aleatorias (300) | vs tirador FSM (duelo, 200) |
| --- | --- | --- | --- |
| **anticipation** | **0.240** [0.19–0.30] | 0.553 [0.50–0.61] | 0.835 [0.78–0.88] |
| Guardia central tras la patada | 0.040 [0.02–0.08] | 0.593 [0.54–0.65] | 0.950 |
| FSM de B-Human | 0.060 | — | 0.905 |
| RL v4 | 0.205 [0.15–0.27] | 0.353 [0.30–0.41] | 0.620 |
| Geométrico | — | 0.213 [0.17–0.26] | — |

**Cómo leerlo:**
- **Contra el tirador v2,** que reparte sus tiros 51/49 entre las dos esquinas, agacharse al centro no sirve (4 %). Leer su postura multiplica las atajadas por 6 y supera al portero FSM y, por poco, al RL v4. El predictor se ajustó con otro rival (`random_kicks`) y aun así funciona contra el v2, porque los dos apuntan con el mismo controlador táctico.
- **Contra patadas aleatorias,** con 25 % de tiros al centro e imprecisas, empata con la guardia central.
- **Contra el tirador FSM,** que patea siempre al centro, se lanza de lado alguna vez de más y pierde ~11 puntos frente a la guardia central.

**Prototipo frente a versión integrada.** Un prototipo se comprometía en el instante exacto del contacto, un dato privilegiado de la simulación. Con patadas aleatorias dio 0.60 frente a 0.57 de la guardia central, con el mejor de 3 umbrales elegido sobre esos mismos 300 penales. La versión integrada detecta la patada por el movimiento del balón, como haría el robot real, y en otras 300 patadas dio 0.55 frente a 0.59. Parte de la diferencia es ruido estadístico (±5–6 puntos por cifra) y parte probablemente el momento de reacción.

## 6. Límites

- **La estirada de B-Human:** aun en el lado correcto ataja 46–59 %, y la izquierda rinde menos que la derecha (0.46 frente a 0.59).
- **Datos privilegiados:** en la simulación, la posición y la orientación del tirador vienen de la escena, con ruido solo en la posición. En el NAO habría que estimarlas con la cámara; la orientación de otro robot es lo más difícil.
- **Depende del controlador del rival:** el predictor aprende cómo se acomoda el controlador táctico de este proyecto. Un tirador con otra forma de apuntar requiere juntar datos y reajustarlo.

## 7. Siguientes pasos

1. **Mejorar la estirada:** medir los 3 movimientos de portero de B-Human que no se usan (sobre todo agacharse con las piernas abiertas) y exponer los `BlockParameters` en el runtime nativo.
2. **Darle al RL la predicción del lado como entrada,** o entrenarlo contra el tirador v2, para que decida cuándo anticipar y cuándo agacharse según el rival.
3. **Elegir el umbral (hoy 0.75) con penales separados** de los de evaluación.

## 8. Cómo usarlo

```powershell
# Reajustar el predictor (junta penales en B-Human)
.\.venv\Scripts\python.exe tools\fit_kick_predictor.py --episodes 600 --workers 30

# Evaluar el portero en las condiciones de su etapa
.\.venv\Scripts\python.exe tools\evaluate_rl.py --role keeper --stage keeper_bhuman_random_kicks --policy anticipation --episodes 100 --seeds 90000 90001 90002 --workers 30

# Duelo contra los tiradores FSM y v2
.\.venv\Scripts\python.exe tools\duel_rl.py --strikers scripted v2=models\rl\striker_v2 --keepers scripted center:gate anticipation --episodes 200 --workers 30

# Verlo en el visor de MuJoCo contra el tirador v2
.\.venv\Scripts\python.exe tools\watch_rl.py --striker models\rl\striker_v2 --keeper anticipation --shots 4 --live
```
