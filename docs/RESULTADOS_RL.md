# Entrenamiento RL: qué se hizo, resultados y dónde está cada cosa

Periodo: 2026-10-08 a 2026-10-10, rama `feat/rl-contracts`. Las corridas ppo_v1 y ppo_v2 se hicieron con el código del commit `132c55c` y la v4 con el commit siguiente (filtro del portero, tiempos sintéticos, duelo, métrica de lados). Modelos y reportes en git en `models/rl/` y `docs/validation/rl/`. Máquina de entrenamiento: Ryzen 9 8940HX (16 núcleos/32 hilos), 31 GB, Windows 11. Diseño y contratos en [RL.md](RL.md).

Todas las cifras son de simulación (MuJoCo + motor B-Human original). Ninguna afirma equivalencia con el NAO real.

## Reentrenamiento contra el portero visual — 2026-10-11

La corrida corta `striker_visual_v3` **no superó al tirador v2**. Se mantiene
`models/rl/striker_v2` como recomendado. El candidato nuevo se conserva en
`models/rl/striker_visual_v3` como experimental, elegido por validación antes
de la prueba independiente. No sustituye a v2 ni cambia su modelo.

Se realizaron 24 actualizaciones PPO y 18 432 transiciones optimizadas
contra nuestro portero visual v2 `balanced`, con movimientos B-Human
originales. La validación eligió el checkpoint 16: 4/32 goles, empatado con
el 24; v2 obtuvo 7/32. La inicialización calibrada sin PPO obtuvo 2/32.
El modelo seleccionado incorpora 12 288 transiciones nuevas; la corrida
completa incorpora 18 432 y conserva los 24 checkpoints.

Prueba de **600 penales nuevos**, 100 por tirador y rival, con semillas y
posiciones iniciales emparejadas dentro de cada rival:

| Rival | Goles v2 | Goles candidato | Diferencia de goles, IC95 pareado |
| --- | ---: | ---: | --- |
| Visual v2 `balanced`, movimiento original | 16/100 | 12/100 | −4 puntos [−12, +4] |
| Visual v2 `balanced`, salto ajustado | 26/100 | 15/100 | −11 puntos [−21, −1] |
| FSM base propia, motor B-Human | 91/100 | 38/100 | −53 puntos [−63, −43] |

Los intervalos de las dos comparaciones secundarias son descriptivos, sin
corrección por comparaciones múltiples. Los goles cuentan todos los penales
programados; cada rival registró un tiro fallado del candidato, contabilizado
aparte de las atajadas. No hubo errores de simulación en validación o prueba.

En los 100 penales ante el portero principal, el candidato solicitó objetivos
más centrados: 53 dentro de ±0.30 m frente a dos de v2; ninguno a ≥0.60 m del
centro frente a 93 de v2. La potencia media solicitada bajó de 0.75 a 0.63.
Son decisiones tácticas observadas, no una prueba de causalidad: el piloto
cambió tanto la inicialización como el entrenamiento PPO.

Hubo un fallo durante la recolección del entrenamiento: se preservó el
checkpoint 4 y se reanudó con semillas nuevas, reservando 774 índices. El
detalle y el volumen físico del rollout descartado no quedaron conservados;
no se afirma cero errores globales ni se equiparan transiciones optimizadas
con todas las transiciones físicas ejecutadas.

El tirador conserva observación privilegiada de escena con ruido; el
portero utiliza RGB y sensores propios ideales simulados. La FSM de control
es del proyecto, no el comportamiento oficial completo de B-Human. No se
aprendieron trayectorias articulares y esta corrida no demuestra convergencia.
Suite: 337 pruebas pasan; paridad física de entorno comprobada en dos pares
fuera de las particiones. Los 559 originales y los modelos anteriores se
conservan intactos.

Método: [ENTRENAMIENTO_TIRADOR_VISUAL.md](ENTRENAMIENTO_TIRADOR_VISUAL.md).
Resultados, auditorías y copia completa de corrida e inputs:
[reporte versionado](validation/rl/striker_visual_v3/README.md).

## Resumen

- **Tirador a usar: v2** (`models/rl/striker_v2`). En B-Human, con variación de posiciones y ruido de percepción, anota el 97 % contra el portero estático y el 96 % contra el portero FSM (100 penales nuevos cada uno). En el duelo contra el portero FSM anota 91.5 % con ruido y 94 % sin ruido; el tirador FSM original, 6–7 %.
- **El v2 reparte los tiros 51 % a su izquierda y 49 % a su derecha**, siempre a la esquina, y anota igual por los dos lados. El v1 patea casi siempre a su derecha, así que es predecible.
- **Portero: ningún entrenado supera a la regla fija "guardia central tras la patada"**, que ataja 0.55. El mejor entrenado (v4) ataja 0.37; el geométrico, 0.18.
- **Portero que lee la postura del tirador (`anticipation`):** contra el tirador v2 ataja **24 %**, frente a 4 % de la guardia central y 6 % del portero FSM. Contra patadas aleatorias empata con la guardia central, y contra el tirador FSM, que patea al centro, rinde algo menos. Ver la sección 4.
- **Portero robusto (`robust`):** en ocho estilos reservados ataja **84.6 %** de media (anticipation 48.4 %, center 45.2 %) y **77 %** en su peor estilo. Con incertidumbre de orientación/velocidad y 100 ms de retraso: 70.2 % de media y 57 % en el peor estilo. Son reglas temporales sobre B-Human, sin reentrenar PPO. Ver [PORTERO_ROBUSTO.md](PORTERO_ROBUSTO.md).
- **El entrenamiento es ~17× más rápido que al inicio**: de ~10 a ~170 transiciones/s en B-Human, con el CPU del 8 % al 100 %. La simulación es idéntica bit a bit.

## 1. Qué se hizo

### 1.1 Puesta en marcha en este PC

- Entorno `.venv` con Python 3.11, torch 2.14.1+cpu, gymnasium 1.4.0, stable-baselines3 2.9.0 y mujoco 3.3.7.
- Smart App Control de Windows bloqueaba `torch\lib\shm.dll` y no dejaba importar torch; se desactivó el 2026-10-08.
- Runtime nativo: este PC solo tiene Visual Studio Build Tools 2026 y `tools/build_motion_runtime.py` fija el generador de VS 2022, así que falla aquí. Se compiló a mano: primero el script (su `prepare()` genera las fuentes adaptadas), después `cmake -S native/bhuman_motion -B native/bhuman_motion/build -G "Visual Studio 18 2026" -A x64` y `cmake --build native/bhuman_motion/build --config Release`. El script no se modificó.
- Modo de energía con corriente: "Máximo rendimiento" (`powercfg /overlaysetactive ded574b5-45a0-4f42-8737-46345c09c238`). Aporta ~15 % de velocidad. Para volver a "Equilibrado": `powercfg /overlaysetactive 00000000-0000-0000-0000-000000000000`.
- La GPU (RTX 5070 Ti) no se usa ni ayudaría. La actualización PPO de 4096 transiciones tarda 0.3 s en CPU, frente a ~28 s de simulación (~1 % del tiempo). La simulación (MuJoCo y el motor B-Human en C++) solo corre en CPU.
- La noche del 2026-10-09 el reloj de Windows se corrigió ~5 h hacia atrás (zona Bogotá). Las horas de ppo_v1 y del inicio de ppo_v2 están en el reloj anterior, y la columna `fps` del log de ppo_v2 es inválida desde ese cambio. Los temporizadores del entrenamiento son monotónicos y no se vieron afectados.

### 1.2 Velocidad (commit 132c55c)

- **Recolección asíncrona** en las etapas B-Human (`penalty/rl/async_ppo.py`, `penalty/rl/actors.py`). Antes, `SubprocVecEnv` avanzaba los 16 entornos sincronizados y cada paso esperaba al más lento: tras una patada se simulan ~440 ticks, frente a ~9 de un paso normal. El CPU quedaba al 8 % y salían ~10 transiciones/s. Ahora cada worker corre su entorno con una copia numpy de la política, y el learner entrena con la actualización PPO de stable-baselines3 sin cambios. Sigue siendo on-policy, y el GAE se corta entre segmentos de workers distintos.
- **Simulación por tick: de 8.0 a 2.34 ms**, idéntica bit a bit en 10 episodios de referencia. Se logró leyendo con búfer los pipes de los motores nativos (antes se leía un byte por llamada), pidiendo los dos motores a la vez, vectorizando los contactos, precalculando constantes y no armando trazas por tick que nadie graba.
- Resultado con 30 workers en "Máximo rendimiento": 148 transiciones/s en el benchmark del tirador y ~170 durante el entrenamiento, con el CPU al 100 % a ~3.2 GHz. En "Equilibrado" salían 121, 122 y 128/s con 16, 24 y 30 workers.

### 1.3 Convergencia y protecciones (commit 132c55c)

- **Convergencia:** las etapas B-Human paran solas; `--hours` es solo un tope. Cada worker juega 1 de cada 5 episodios con la acción media, sin ruido de exploración, y esos episodios forman ventanas de 200. Una etapa converge con 4 ventanas seguidas sin mejora y al menos 60 min.
- **Mejor checkpoint:** se guarda en `<rol>/<etapa>_best` y se restaura si al cerrar la última ventana es claramente peor.
- **Protecciones:** se aborta la corrida si `sim_error` supera el 10 %. Si aparecen NaN, no se sobrescribe el checkpoint bueno. Un watchdog termina el proceso si pasan 15 min sin pasos. Un supervisor (`tools/train_rl.py`) reanuda con `--resume` tras un fallo, hasta 3 seguidos.

### 1.4 Condiciones de entrenamiento (commit 132c55c)

- **Ruido de percepción** en todas las etapas.
- **Variación en B-Human:** ±5 cm en la posición del balón y ±5 cm / ±0.1 rad en la pose inicial de los robots. Sin esta variación, el mundo del tirador es determinista: la semilla no cambia el episodio.
- **Patadas a las esquinas para el portero:** 3 de cada 4 apuntan a |y| entre 0.3 y 0.65 m, 1 de cada 4 al centro, con alcance de 1960–2900 mm.
- **Exploración controlada en B-Human:** sin bono de entropía y desviación ≤ 0.5. En v1 había crecido hasta 0.7–20.
- **Baseline adicional del portero:** `center` (guardia central), además del geométrico.

### 1.5 Portero: filtro y tiempos (2026-10-10, sin commit)

- **Filtro `commit_after_shot`** (sintético y B-Human): una guardia o estirada pedida antes de que el rival toque el balón se ignora (`info['safety'] = 'waiting_for_shot'`).
- **Tiros sintéticos a los 3–9 s** (`kick_time_range`), como en B-Human. Antes eran 0.5–3 s.
- **Desviación ≥ 0.2** en B-Human, para que no deje de explorar.

### 1.6 Duelo entre políticas (2026-10-10, sin commit)

`penalty/rl/duel.py` y `tools/duel_rl.py` enfrentan tiradores y porteros, entrenados o baselines. Los dos deciden cada 0.1 s con su propia observación y su propio ruido. Todas las parejas juegan las mismas semillas, así que la comparación es pareada.

## 2. Corridas

| Corrida | Qué cambió | Duración | Resultado | Carpeta |
| --- | --- | --- | --- | --- |
| ppo_v1 | Ambos roles; sin variación ni ruido; exploración libre | 7 h 16 min (incluye ~1 h detenida para optimizar) | Tirador 24/24 en un escenario fijo; portero 0.778 agachándose antes del tiro | `runs/results/ppo_v1_20261009/` |
| ppo_v2 | Ambos roles; variación, ruido, patadas a las esquinas, convergencia sin ruido | ~4 h | Tirador 0.97 / 0.96; portero 0.13 (se cae antes del tiro) | `runs/results/ppo_v2_20261010/` |
| ppo_v3_keeper | Portero; solo tiempos de 3–9 s | Detenida a los ~25 min | Sintético 0.20: no aprende a esperar | `runs/train/ppo_v3_keeper_20261010/` |
| ppo_v4_keeper | Portero; filtro, tiempos y desviación ≥ 0.2 | ~1 h 20 min | 0.37, frente a 0.18 del geométrico y 0.55 de la guardia central | `runs/results/ppo_v4_keeper_20261010/` |

Las etapas sintéticas de todas las corridas duran pocos minutos y sirven de calentamiento. Lo que cuenta son las etapas B-Human.

### ppo_v1

| Etapa B-Human | Entrenado (24 penales) | Baseline |
| --- | --- | --- |
| Tirador vs portero estático | 1.000 | scripted 0.000 |
| Tirador vs portero FSM | 1.000 | scripted 0.000 |
| Portero vs patadas uniformes | 0.792 | geométrico 0.417 |

- **Evaluación formal del portero** (5 semillas × 100): 0.778 [0.740–0.812] frente a 0.420 [0.378–0.464] del geométrico.
- **Lo que aprendió el portero:** se compromete a la guardia central antes de cada tiro (estirada temprana 1.0, lado equivocado 0.55, sin caídas). Es una estrategia fija que no lee el tiro.
- **Lo que aprendió el tirador:** como el mundo era determinista, 24/24 es el mismo tiro repetido.
- **Con la variación y el ruido de ppo_v2** (100 penales): el tirador v1 baja a 0.81 y el portero v1 a 0.09, con 85 % de caídas.

### ppo_v2

| Etapa B-Human | Entrenado (100 penales) | Baselines |
| --- | --- | --- |
| Tirador vs portero estático | **0.970** [0.92–0.99] | scripted 0.310 |
| Tirador vs portero FSM | **0.960** [0.90–0.98] | scripted 0.020 |
| Portero vs patadas a las esquinas | 0.130 [0.08–0.21] | geométrico 0.180, center 0.460 |

El portero falló por tres razones:
- **Tiempos distintos:** el mundo sintético lo entrenó con tiros a los 0.5–3 s, pero en B-Human el rival camina hasta el balón y patea a los 3.1–7.4 s (p10–p90, mediana 5.7 s, 60 episodios). Se lanzaba en la primera o segunda decisión y caía antes del tiro (`keeper_fall` 0.79).
- **Casi sin actualizaciones:** con episodios de ~3 decisiones, cada lote de 4096 transiciones exigía ~1300 reinicios de B-Human. Solo hubo 27 actualizaciones PPO en la hora de la etapa.
- **Sin exploración:** llegó del sintético con una desviación del gatillo de 0.13. La etapa no mejoró nunca y al cerrar se restauró la primera ventana.

### ppo_v3_keeper (detenida)

Solo cambió los tiros sintéticos a 3–9 s. El portero tampoco aprendió a esperar las 30–90 decisiones previas al tiro: 0.204 de atajadas en la primera etapa sintética, y 0.13–0.20 con ~60 % de caídas tempranas tras 3 M pasos. Se detuvo antes de llegar a B-Human.

### ppo_v4_keeper

| Etapa | Entrenado | Baselines |
| --- | --- | --- |
| Sintético fácil | 0.850 | geométrico 0.980, center 0.462 |
| Sintético completo | 0.618 | geométrico 0.690, center 0.382 |
| B-Human vs patadas a las esquinas (100 penales) | **0.37** [0.28–0.47] | geométrico 0.18 [0.12–0.27], **center 0.55** [0.45–0.64] |

Ya no se cae antes del tiro y duplica al geométrico. Pero casi siempre se lanza de lado (caídas 0.87, que aquí son las estiradas después del tiro) y elige el lado equivocado 2 de cada 3 veces. Agacharse al centro después de la patada ataja más.

## 3. Duelo: tiradores contra el portero FSM de B-Human

200 penales por tirador con las mismas semillas, 2026-10-10. Reporte: `runs/results/duelo_tiradores_vs_portero_fsm_20261010.json`.

| Tirador | Goles sin ruido | Goles con ruido |
| --- | --- | --- |
| FSM original | 5.5 % [3–10] | 7.0 % [4–11] |
| v1 | 92.0 % [87–95] | 75.0 % [69–81] |
| **v2** | **94.0 %** [90–97] | **91.5 %** [87–95] |

- **Entrenados frente al FSM:** el FSM patea al centro (cruza a ~0.27 m) a los 3.0 s. Los entrenados rodean el balón y patean a las esquinas (~0.55 m) a los 6.0–6.5 s.
- **v2 frente a v1 sin ruido:** 5 penales que solo anota v2 contra 1 que solo anota v1 (z = 1.6). La diferencia no es significativa.
- **v2 frente a v1 con ruido:** 39 penales que solo anota v2 contra 6 que solo anota v1 (z = 4.9). La diferencia de 16.5 puntos sí es significativa: el v1 se entrenó sin ruido y pierde 17 puntos cuando lo hay.

### Hacia qué lado patea

Métrica `sides` de `penalty/rl/metrics.py`, calculada sobre los mismos penales del duelo:
- Todo está en la vista del tirador: **su izquierda es la derecha del portero**.
- Un tiro va a un lado si |y| > 0.2 m en la línea de gol; si no, va al centro.
- "Apunta" es el objetivo que elige la política; "cruza" es por dónde pasa el balón la línea de gol, cuando la pasa.

| Tirador (condición) | Apunta izq. / centro / der. | Cruza la línea izq. / centro / der. | Gol si apunta izq. / der. | \|y\| medio apuntado |
| --- | --- | --- | --- | --- |
| FSM (sin ruido) | 0 / 100 / 0 % | 9 / 55 / 36 % (solo 11 cruzan) | — (6 % al centro) | 0.00 m |
| v1 (sin ruido) | 0 / 1 / 99 % | 1 / 0 / 99 % | — / 92 % | 0.48 m |
| v1 (con ruido) | 0 / 16 / 84 % | 0 / 0 / 100 % | — / 83 % (centro 34 %) | 0.38 m |
| **v2 (sin ruido)** | **51 / 0 / 49 %** | 51 / 0 / 49 % | **94 / 94 %** | 0.64 m |
| **v2 (con ruido)** | **51 / 0 / 49 %** | 51 / 0 / 49 % | **91 / 92 %** | 0.64 m |

- **v1** patea casi siempre a su derecha. Un portero que lo sepa se tiraría siempre a su izquierda.
- **v2** reparte los tiros mitad y mitad, siempre a la esquina (0.64 m de un máximo de 0.65), y anota igual por los dos lados.

Desde ahora `summarize` agrega `sides` en toda evaluación del tirador, y `tools/duel_rl.py` en cada pareja. Los registros de B-Human y del sintético guardan `target_y` y `goal_line_y`.

### Videos

En `docs/media/rl/` hay 6 penales del duelo con la condición realista (variación y ruido), reproducidos con la misma semilla y con el mismo resultado. La cámara está detrás y arriba del tirador: **la izquierda de la imagen es la izquierda del tirador**.

| Video | Tirador vs portero | Lado | Resultado |
| --- | --- | --- | --- |
| `tiro1_v2_vs_scripted_derecha_goal.mp4` | v2 vs FSM | derecha (−0.65 m) | gol, cruza en −0.57 m |
| `tiro2_v2_vs_scripted_izquierda_goal.mp4` | v2 vs FSM | izquierda (+0.65 m) | gol, cruza en +0.52 m |
| `tiro3_v2_vs_scripted_derecha_goal.mp4` | v2 vs FSM | derecha | gol, cruza en −0.61 m |
| `tiro4_v2_vs_scripted_izquierda_goal.mp4` | v2 vs FSM | izquierda | gol, cruza en +0.54 m |
| `tiro5_v2_vs_scripted_izquierda_save.mp4` | v2 vs FSM | izquierda | atajada (pasa en el 6 % de sus tiros) |
| `tiro1_scripted_vs_scripted_centro_save.mp4` | FSM vs FSM | centro | atajada |

Para verlos en vivo en el visor de MuJoCo, o para grabar otros, usa `tools/watch_rl.py` (ver Comandos).

## 4. Portero que lee al tirador (anticipación)

Documento completo: [MEJORA_DE_PORTERO.md](MEJORA_DE_PORTERO.md).

Para apuntar fuera del centro, el tirador táctico rodea el balón, así que su postura antes de patear delata el lado. El portero ya recibía la posición y la orientación del rival en su observación, pero el RL nunca aprendió a usarlas: en el mundo sintético el rival está quieto y no da pistas.

**Se puede predecir el lado.** En 599 penales de B-Human (patadas a las esquinas con variación), un clasificador sobre la postura del tirador, tal como la ve el portero, acertó:

| Momento | Acierto (3 lados) | Solo tiros a las esquinas |
| --- | --- | --- |
| Al patear | 77 % | 87 % |
| 1 s antes | 77 % | 85 % |
| 2 s antes | 74 % | 82 % |

El balón termina en el lado apuntado solo el 73 % de las veces: las patadas de B-Human son imprecisas.

**Anticipar tiene un techo.** Un oráculo que conoce el lado real y se compromete en la primera decisión tras la patada ataja 0.62, frente a 0.57 de la guardia central. Aun en el lado correcto, las estiradas de B-Human atajan 0.59 (derecha del portero) y 0.46 (izquierda); la guardia central ante tiros al centro, 0.83. El límite es la estirada.

**Componentes:**
- `penalty/rl/anticipation.py`:
  - *Variables de la postura:* posición del tirador respecto al balón, su orientación y el punto donde la línea de su mirada cruza la línea del portero.
  - *Predictor:* regresión logística de 3 clases.
  - *`AnticipatingKeeperPolicy`:* espera a que el balón se mueva y entonces se lanza al lado predicho si la probabilidad es ≥ 0.75; si no, se agacha al centro. No depende del filtro `commit_after_shot`.
- `tools/fit_kick_predictor.py` genera `models/rl/kick_side_predictor.json`: 566 penales, acierto en validación cruzada de 78 % (86 % en las esquinas).
- El portero `anticipation` está en `make_policy` y en los baselines del portero.

**Resultados** (`docs/validation/rl/anticipacion_20261010/`, penales que el predictor nunca vio, con ruido):

| Portero | Patadas aleatorias (300) | vs tirador v2 (duelo, 200) | vs tirador FSM (duelo, 200) |
| --- | --- | --- | --- |
| **anticipation** | 0.553 [0.50–0.61] | **0.240** [0.19–0.30] | 0.835 [0.78–0.88] |
| Guardia central tras la patada | 0.593 [0.54–0.65] | 0.040 [0.02–0.08] | 0.950 |
| FSM de B-Human | — | 0.060 | 0.905 |
| RL v4 | 0.353 [0.30–0.41] | 0.205 [0.15–0.27] | 0.620 |
| Geométrico | 0.213 [0.17–0.26] | — | — |

- **Contra el tirador v2, que patea siempre a las esquinas**, leer su postura multiplica las atajadas por 6 respecto a la guardia central, y supera al portero FSM y, por poco, al RL v4.
- **Contra patadas aleatorias** (25 % al centro e imprecisas) empata con la guardia central.
- **Contra el tirador FSM**, que patea siempre al centro, se lanza de lado alguna vez de más y pierde ~11 puntos.
- **Momento de reacción:** un prototipo que se comprometía en el instante exacto del contacto (dato privilegiado de la simulación) dio 0.60 frente a 0.57 con patadas aleatorias; la versión integrada detecta la patada por el movimiento del balón, como haría el robot real. El umbral 0.75 se eligió con el prototipo.

## 5. Hallazgos

- En esta simulación **agacharse al centro rinde más que lanzarse**. Las patadas B-Human son imprecisas (con apuntado a las esquinas, 21 % salen fuera) y la estirada lateral rinde poco. Contra patadas a las esquinas, con variación y ruido:

  | Estrategia del portero | Atajadas |
  | --- | --- |
  | Guardia central tras la patada | 0.55 |
  | Guardia central inmediata | 0.48 |
  | No comprometerse nunca (el cuerpo bloquea) | 0.28 |
  | Geométrico | 0.18–0.28 |

- **El mundo sintético del portero tiene que imitar los tiempos de B-Human**; si no, la política no se transfiere.
- **Esperar decenas de decisiones sin "disparar" es muy difícil para PPO** con ruido por paso. El filtro `commit_after_shot` lo resuelve.
- **Con acciones recortadas a [-1, 1], el bono de entropía infla la exploración:** los episodios de entrenamiento se vuelven casi aleatorios (en v1, 0.26 de atajadas al entrenar contra 0.79 sin ruido).
- **Medir la convergencia con episodios de entrenamiento engaña** cuando hay mucho ruido de exploración; por eso ahora se mide sin ruido.

## 6. Dónde está cada cosa

### Código (en git)

| Archivo | Contenido |
| --- | --- |
| `penalty/rl/training.py` | Currículo (`overnight_plan`; condiciones `NOISE`, `VARIATION`, `CORNER_KICKS`, `KEEPER_TIMING`, `BHUMAN_EVAL`), `PPO_SETTINGS`, convergencia (`ConvergenceMonitor`), protecciones (`check_health`, `Watchdog`), baselines de cada etapa (`BASELINES`) |
| `penalty/rl/async_ppo.py`, `actors.py` | Recolección asíncrona: `AsyncPPO`, `AsyncCollector`, workers con política numpy |
| `penalty/rl/duel.py`, `tools/duel_rl.py` | Duelo entre políticas |
| `tools/watch_rl.py` | Ver penales en vivo en el visor de MuJoCo o grabarlos en MP4 (necesita `imageio[ffmpeg]`) |
| `penalty/rl/metrics.py` | Métricas, incluida `sides` (hacia qué lado patea el tirador) |
| `penalty/rl/anticipation.py`, `tools/fit_kick_predictor.py` | Portero `anticipation` y su predictor del lado (`models/rl/kick_side_predictor.json`) |
| `penalty/rl/bhuman_world.py` | Variación, distribución de patadas y filtro del portero en B-Human |
| `penalty/rl/synthetic.py` | Filtro y tiempos del portero sintético |
| `penalty/rl/wrappers.py`, `evaluation.py` | `PerceptionNoise`; opción `perception_noise` de los entornos |
| `penalty/rl/policies/baselines.py` | Baselines, incluido `center` |
| `penalty/bhuman_bridge.py`, `physical_sim.py`, `sim_session.py` | Optimizaciones bit a bit y `start_offsets` |
| `tools/train_rl.py`, `tools/evaluate_rl.py` | Supervisor del entrenamiento; evaluación con `--stage` |
| `tests/test_rl_training.py`, `tests/test_rl_bhuman.py` | Pruebas de lo anterior (la suite completa tiene 109 y pasa en ~70 s) |

### Modelos, reportes y videos (en git)

| Ruta | Contenido |
| --- | --- |
| `models/rl/` | Políticas finales `striker_v1`, `keeper_v1`, `striker_v2`, `keeper_v2` y `keeper_v4` (`model.zip`, `obs_norm.npz`, `config.json`). `manifest.json` indica la corrida, la etapa, el estado, la evaluación y si el portero necesita `commit_after_shot` (`:gate`). |
| `docs/validation/rl/<corrida>/` | `report.json` (resultados por etapa, ventanas, procedencia), `resumen.json` y `curvas/<etapa>.csv` (curvas de entrenamiento). ppo_v1 incluye `eval/` (portero 5×100, tirador vs estático). |
| `docs/validation/rl/ppo_v3_keeper_20261010_detenida/` | `progress.json` de la corrida detenida |
| `docs/validation/rl/duelo_tiradores_vs_portero_fsm_20261010.json` | Duelo: resumen por pareja con `sides` y el registro de cada penal |
| `docs/validation/rl/calibracion_20261010/` | Velocidad por número de workers y pasos por episodio |
| `docs/validation/rl/anticipacion_20261010/` | Evaluación del portero `anticipation` y de los baselines (3×100) y duelo de porteros contra los tiradores FSM y v2 |
| `docs/media/rl/` | Videos de los penales |

### Copias completas (solo en este PC: `runs/` no está en git)

Incluyen todos los checkpoints y los logs de consola.

| Ruta | Contenido |
| --- | --- |
| `runs/results/ppo_v1_20261009/` | `run/`, `logs/` (parte sincronizada y asíncrona), `eval/` (portero 5×100, tirador vs estático), `resumen.json`, `codigo_vs_9682fe0.patch` |
| `runs/results/ppo_v2_20261010/` | `run/`, `logs/`, `resumen.json`, `codigo.txt` |
| `runs/results/ppo_v4_keeper_20261010/` | `run/`, `logs/`, `resumen.json` |
| `runs/results/duelo_tiradores_vs_portero_fsm_20261010.json` | Duelo con los registros de cada penal |
| `runs/results/calibracion_20261010/` | Benchmark de velocidad, pasos por episodio, log de ppo_v3 |
| `runs/train/` | Corridas originales, pruebas cortas (`smoke_*`) y `backup_24h_20261009_1917` (v1 antes de optimizar) |

Dentro de cada `run/`:
- `report.json`: resultados por etapa, ventanas de convergencia, reanudaciones y procedencia.
- `<rol>/latest`, `<rol>/<etapa>` y `<rol>/<etapa>_best`: cada una con `model.zip`, `obs_norm.npz` y `config.json`.
- `logs/<etapa>/progress.csv`: curvas de entrenamiento.

### Comandos

```powershell
# Entrenar (tope de horas; para por convergencia; se reanuda solo tras fallos)
.\.venv\Scripts\python.exe tools\train_rl.py --hours 48 --workers 30
.\.venv\Scripts\python.exe tools\train_rl.py --roles keeper --hours 48 --workers 30 --output runs\train\<nombre>
.\.venv\Scripts\python.exe tools\train_rl.py --output runs\train\<nombre> --hours 48 --workers 30 --resume

# Evaluar en las condiciones de una etapa
.\.venv\Scripts\python.exe tools\evaluate_rl.py --role striker --stage striker_bhuman_scripted --policy models\rl\striker_v2 --episodes 100 --seeds 20000 20001 20002 20003 20004 --workers 30

# Duelo
.\.venv\Scripts\python.exe tools\duel_rl.py --strikers scripted v2=models\rl\striker_v2 --keepers scripted v4=models\rl\keeper_v4:gate --episodes 200 --workers 30 --conditions realista sin_ruido

# Ver penales en vivo (visor de MuJoCo) o grabarlos (un MP4 por tiro)
.\.venv\Scripts\python.exe tools\watch_rl.py --striker models\rl\striker_v2 --keeper scripted --shots 4 --live
.\.venv\Scripts\python.exe tools\watch_rl.py --striker models\rl\striker_v2 --keeper scripted --seeds 50000000 50000002 --video runs\videos
```

## 7. Pendiente

- **Mejorar la estirada**, que es el límite del portero: aun en el lado correcto ataja 0.46–0.59. Opciones: probar los otros 3 movimientos de portero de B-Human que no se usan (`squatArmsBackRight` y `squatWideArmsBackLeft/Right`, agacharse con las piernas abiertas) y exponer los `BlockParameters` en el runtime nativo.
- **Dar al RL la predicción del lado** como entrada, o entrenarlo contra el tirador v2, para que combine anticipación y guardia central según el rival.
- **Duelo de los tiradores contra los porteros entrenados** (v1 y v4) y contra `center`.
- **`tools/build_motion_runtime.py` no detecta VS 2026.**
- Sin cambios respecto al plan P1: self-play, residual RL y adaptador NAO.

## 8. Robustez frente a distintos estilos

Se seleccionó una política temporal con cinco estilos y semillas apartadas;
después se congelaron los parámetros y se jugaron 4800 episodios reservados:
ocho estilos, tres porteros y dos condiciones de percepción, 100 por pareja.
Los estilos incluyen objetivos centrales/mixtos/esquinas, tiros suaves,
disparos demorados, cambios de objetivo y las políticas v1/v2.

`robust` confirma avance del balón, combina proyecciones recientes y activa
el salto lateral cuando la llegada prevista está a ≤1.8 s. La guardia central
se mantiene para cruces dentro de ±0.30 m. La postura previa resuelve solo
ambigüedades cerca del borde. La política conserva los contratos v1 y no usa
el contacto privilegiado ni la identidad del oponente.

En la condición original, la ganancia media frente a anticipation es de
36.25 puntos (IC95 pareado 31.63–40.88); en la condición degradada, de 54.75
puntos (49.50–59.88). Contra el tirador central, robust y center hacen 89/100;
contra v2, robust hace 90/100 y anticipation 27/100. Los movimientos originales
permiten superar el ensayo anterior ajustando el momento de activación.

La condición original contiene cinco errores nativos del tirador entre 2400
episodios; se conservaron en el denominador y se diagnosticaron por separado.
La condición degradada no contiene errores, pero robust conserva 26/800
compromisos marcados tempranos. El ruido sigue siendo analítico sobre poses
de escena; estos resultados no validan percepción ni movimientos en NAO real.

Configuración: `models/rl/keeper_robust_v1.json`. Reportes completos, resumen
pareado y fuentes de recolección: `docs/validation/rl/robustez_20261010/`.
El comando `tools/watch_rl.py --keeper robust` permite ver la política.

## Integración de cámara — 2026-10-10

Primera etapa ejecutable: `vision_keeper_v2`, reglas temporales que reciben
RGB desde una cámara en la cabeza de MuJoCo. Estima posición/velocidad con
calibración y odometría, confirma capturas independientes y no usa orientación
del tirador. `rl_obs_v2` conserva máscaras explícitas; v1 sigue intacto.

Ensayo preliminar con semillas `65001000000`–`65001000009`, 10 tiros por cada
uno de los ocho estilos y portero: 74/80 atajadas (92.5 %) frente a 44/80
(55 %) del base con la misma visión. Ganancia pareada 37.5 puntos, IC95 25–50;
cero errores y cero compromisos marcados tempranos. No se compara directamente
con el 84.6 % anterior: cambian semillas y percepción. La calibración,
odometría y extrínsecos de esta prueba son ideales; no hay resultado físico.

La ruta NAOqi incorpora cámara, tiempos y decisiones en modo de observación;
el ejecutor de bloqueos sigue pendiente. El adaptador consume snapshots de
BallModel/ObstacleModel, pero los proveedores de percepción B-Human todavía
no están activos en este host. Reporte y trazas:
`docs/validation/rl/vision_20261010/`; uso y límites en
[VISION_PORTERO.md](VISION_PORTERO.md).
