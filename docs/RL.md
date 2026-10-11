# RL jerárquico sobre B-Human

Implementación de la base P0 del paquete de diseño en [`docs/rl/`](rl/00_README.md). B-Human sigue siendo el controlador de bajo nivel: la política táctica decide **qué** hacer y nunca escribe los 23 servos, raíces ni velocidad del balón.

```text
L3 táctica (RL o baseline)  -> StrikerAction / KeeperAction
L2 skills parametrizadas    -> ParameterizedKick / ParameterizedBlock (límites duros)
L1 B-Human                  -> FSM compartida + MotionEngine original en MuJoCo
```

## Instalación

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
```

## Módulos

| Ruta | Contenido |
| --- | --- |
| `penalty/rl/contracts.py` | `RLObservation` (`rl_obs_v1`, 19 campos), `StrikerAction`, `KeeperAction`, codificación normalizada `[-1, 1]` y límites. |
| `penalty/rl/geometry.py` | Geometría de `sim/scene.xml`, transformaciones al marco del robot y `predict_goal_crossing`. |
| `penalty/rl/observation.py` | `WorldState -> RLObservation` relativo al robot; velocidad del balón por diferencias finitas, `ball_age` y máscara `ball_visible`. |
| `penalty/rl/wrappers.py` | `ObservationHistory(length=6)`, `DelayedObservationWrapper`, `PerceptionNoiseWrapper` (ruido + pérdida visual). |
| `penalty/rl/controllers.py` | `TacticalStrikerController` (SELECT_SHOT + filtro de seguridad + pose de patada), `TacticalKeeperController`, `StaticKeeperController`. |
| `penalty/rl/striker_env.py`, `keeper_env.py`, `envs.py` | `PenaltyStrikerEnv` y `PenaltyKeeperEnv` compatibles con Gymnasium. |
| `penalty/rl/synthetic.py` | Mundo analítico para currículo etapa 1 y pruebas rápidas. |
| `penalty/rl/bhuman_world.py` | Mundo B-Human/MuJoCo sobre `SimulationSession`. |
| `penalty/rl/rewards.py`, `metrics.py` | `striker_reward_v1`, `keeper_reward_v1`, tasas con IC95 de Wilson; `sides` del tirador (hacia dónde apunta y por dónde cruza la línea de gol). |
| `penalty/rl/policies/baselines.py` | Baselines scripted (equivalentes a la FSM), geométrica de portero y aleatoria. |
| `penalty/rl/evaluation.py`, `tools/evaluate_rl.py` | Benchmark multi-semilla, multiproceso, throughput e informe con procedencia. |
| `penalty/rl/run_metadata.py` | Commit, estado sucio, versiones, hashes y plataforma de cada ejecución. |
| `penalty/rl/training.py`, `tools/train_rl.py` | Currículo PPO, convergencia, protecciones, checkpoints y supervisor con reinicio. |
| `penalty/rl/async_ppo.py`, `actors.py` | Recolección asíncrona para B-Human: workers con política numpy y `AsyncPPO`. |
| `penalty/rl/duel.py`, `tools/duel_rl.py`, `tools/watch_rl.py` | Duelo entre políticas; ver penales en el visor de MuJoCo o grabarlos en MP4. |
| `penalty/rl/anticipation.py`, `tools/fit_kick_predictor.py` | Portero que lee la postura del tirador: predictor del lado (`models/rl/kick_side_predictor.json`) y baseline `anticipation`. Detalle en [MEJORA_DE_PORTERO.md](MEJORA_DE_PORTERO.md). |
| `penalty/rl/robust_keeper.py`, `keeper_benchmark.py`, `keeper_noise.py`, `tools/evaluate_keeper_robustness.py` | Baseline temporal `robust`, benchmark de ocho estilos y perturbaciones versionadas. 84.6 % de atajadas de media; peor estilo 77 %. Con perturbaciones: 70.2 % y 57 %. Ver [PORTERO_ROBUSTO.md](PORTERO_ROBUSTO.md). |
| `penalty/robot_detector.py`, `opponent_perception.py`, `sim_opponent_vision.py`, `rl/opponent_benchmark.py` | Detector B-Human activo desde imágenes superiores/inferiores; posición aproximada del rival y orientación desconocida, con máscaras `rl_obs_v2`. Nuevo ensayo separado mediante `--opponent-detector bhuman`. Ver [VISION_BHUMAN_RIVAL.md](VISION_BHUMAN_RIVAL.md). |
| `penalty/opponent_orientation.py`, `sim_sensors.py`, `sim_orientation.py`, `keeper_pose_vision.py`, `rl/orientation_contracts.py`, `rl/orientation_benchmark.py` | Orientación corporal desde RGB, cámara reconstruida con encoders/IMU/FSR, `vision_state_v2`/`rl_obs_v3` y política visual v3. Selección nueva `--opponent-detector bhuman --orientation visual`; validación limitada a apariencia simulada. Ver [ORIENTACION_PORTERO.md](ORIENTACION_PORTERO.md). |
| `penalty/keeper_fusion.py`, `rl/fusion_benchmark.py`, `tools/evaluate_keeper_fusion.py`, `tools/fit_kick_heading.py`, `tools/analyze_keeper_fusion.py` | Porteros v4 (v2 + orientación congelada al tiro) y v5 (recta de mínimos cuadrados, centro tardío y límites por lado; `ball_v5` sin orientación). Revisión de v3, desarrollo y experimento final pareado en [PORTERO_FUSION.md](PORTERO_FUSION.md). |
| `models/rl/`, `docs/validation/rl/`, `docs/media/rl/` | Políticas entrenadas (con `manifest.json`), reportes y curvas, y videos. |
| `penalty/skills/` | `KickParameters`, `BlockParameters`, `ParameterizedSkill` (`reset/apply/bounds`). |
| `penalty/optimization/` | Búsqueda aleatoria acotada, CMA-ES, objetivos `kick_objective_v1`/`block_objective_v1`, `kick_search`. |

## Contratos

**Observación** `rl_obs_v1`: los 18 campos canónicos del diseño más `ball_visible`. Posiciones en metros en el marco del robot (x adelante, y izquierda); ángulos en radianes. Los postes se nombran como los ve el tirador mirando la portería. `remaining_time` usa el límite de 28 s.

**Acción del tirador** `Box(-1, 1, 4)`: `target_y` (±0.65 m sobre la línea de gol, positivo a la izquierda del tirador), `kick_side` (izquierdo si ≤ 0), `kick_power` y `trigger`. La patada solo sale si `trigger ≥ 0.5`, la FSM está alineada con el punto elegido y el filtro de seguridad acepta: balón ≤ 0.25 m y |bearing| ≤ 0.15 rad, motor no ocupado, inclinación ≤ 12° y |gyro| ≤ 0.4 rad/s. `info['safety']` informa el motivo del rechazo.

**Acción del portero** `Box(-1, 1, 3)`: lado (`<-1/3` derecha, `>1/3` izquierda, centro en medio), `trigger` (`WAIT` si `< 0.5`) y `amplitude`. El lado se compromete una vez.

Cada `reset()` devuelve en `info` las versiones de observación/acción/recompensa, backend, oponente y las entradas privilegiadas o sintéticas.

## Backends

### `synthetic`

Modelo analítico (rodadura con desaceleración constante, cobertura de estirada simplificada) sin MuJoCo. Sirve para la etapa *synthetic trajectories* del currículo del portero, depuración y pruebas a miles de episodios/s. **No es física**: todo resultado debe reevaluarse en `bhuman`.

Oponentes del tirador: `empty`, `static`, `random`, `scripted`. Portero: `synthetic` (tiros muestreados con `speed_range`, `target_range`, `kick_time_range` configurables vía `backend_options` u `options` de `reset`).

### `bhuman`

Un paso táctico es un periodo de control de 0.1 s de `SimulationSession`. La aproximación, la marcha, la patada y la guardia siguen en la FSM compartida y B-Human. Tras liberar la patada o comprometer la guardia, el entorno avanza hasta el resultado del balón (máximo 10 s).

- Tirador: oponentes `scripted` (portero FSM) y `static` (solo sigue el balón).
- Portero: oponentes `scripted` (tirador FSM, centro, pie izquierdo, 2500 mm) y `random_kicks` (patadas B-Human con objetivo, pie y alcance aleatorios por episodio). Por defecto el objetivo es uniforme en ±0.65 m; `shot_target=(min, max)` sortea |objetivo| en ese rango con lado aleatorio, salvo una fracción `center_shots` dentro de ±0.15 m, y `shot_power=(min, max)` acota el alcance.
- Variación por episodio (`backend_options`): `ball_jitter` (± m alrededor del punto penal) y `start_jitter=(± m, ± rad)` de ambos robots respecto a su pose inicial nominal. Sin ellas el mundo del tirador es determinista: la semilla no cambia el episodio.
- `perception_noise` (`True` o argumentos de `PerceptionNoiseWrapper`) en `backend_options` envuelve el entorno con ruido de percepción, en entrenamiento y evaluación por igual.
- Con valores por defecto la simulación es bit a bit la de antes de estas opciones (comprobado en 10 episodios con semilla).
- Pose de patada: para apuntar fuera del centro, el tirador táctico se mueve sobre el círculo alrededor del balón. Girar en el sitio a 0.2 m del balón lo empuja con los pies. Las compuertas de disparo son las de la FSM, así que el objetivo centrado reproduce el penal scripted (`blocked`).
- Un fallo nativo/físico termina el episodio como `sim_error`, puntuado como caída, y descarta la sesión.

Entradas privilegiadas declaradas: posición del balón y poses de los robots desde la escena, orientación del torso. Envolver con `PerceptionNoiseWrapper` y `DelayedObservationWrapper` para las fases B/C de sim-to-real.

## Recompensas y terminaciones

`striker_reward_v1`: +10 gol, −8 bloqueado, −6 desviado/fuera/corto, −10 caída (incluye `sim_error`), −0.02 por paso. Terminaciones `goal`, `blocked`, `wide`, `out`, `short`, `fall`, `sim_error`; `timeout` trunca.

`keeper_reward_v1`: +10 atajada, +4 contacto, −10 gol, −5 caída sin atajada, −2 estirada temprana (antes del tiro o > 1 s antes de la llegada), −0.02 por paso. Terminaciones `save`, `goal`, `out`, `keeper_fall` (caída antes del tiro), `sim_error`; `timeout` trunca.

No editar una recompensa publicada: crear `*_v2`.

## Skills parametrizadas

El KickEngine original expone hoy solo el pie y el alcance pedido (`kick_length_mm`). `power ∈ [0, 1]` se mapea a 1500–2900 mm. El 2026-10-07 se sondeó en MuJoCo, con ambos pies y portero estático: 1500–2900 mm terminan limpios; 3100 mm lleva una articulación del tirador por encima del límite de 100 rad/s del runtime. El valor nominal (2500 mm) es `NOMINAL_POWER`.

`swing_amplitude`, `hip_rotation`, `foot_angle`, `contact_phase`, `support_lean` y todos los `BlockParameters` tienen límites y versión, pero `apply()` **rechaza** valores no nominales hasta que una adaptación generada en `native/bhuman_motion/build/generated` los exponga. `bounds()` devuelve solo lo optimizable. `amplitude` del portero se informa en `info['unapplied']` con B-Human.

## Comandos

```powershell
# Pruebas RL (las de B-Human se omiten si el runtime no está compilado)
.\.venv\Scripts\python.exe -m unittest tests.test_rl_contracts tests.test_rl_envs tests.test_rl_bhuman -v

# Baseline: >=5 semillas, >=100 episodios por configuración
.\.venv\Scripts\python.exe tools/evaluate_rl.py --role keeper --policy geometric --episodes 100 --seeds 0 1 2 3 4
.\.venv\Scripts\python.exe tools/evaluate_rl.py --role striker --backend bhuman --episodes 20 --seeds 0 1 2 3 4 --workers 4 --output runs/baseline_report_striker.json

# Paralelismo real con B-Human
.\.venv\Scripts\python.exe tools/evaluate_rl.py --role striker --backend bhuman --episodes 4 --throughput 1 4 8 12

# Optimización del alcance de patada contra un objetivo fijo
.\.venv\Scripts\python.exe -m penalty.optimization.kick_search --method cmaes --budget 12 --target-y 0.4
```

Uso con cualquier biblioteca Gymnasium:

```python
from penalty.rl.keeper_env import PenaltyKeeperEnv
from penalty.rl.wrappers import ObservationHistory, PerceptionNoiseWrapper
env = ObservationHistory(PerceptionNoiseWrapper(PenaltyKeeperEnv(backend='synthetic')), length=6)
```

## Entrenamiento PPO hasta convergencia

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-rl.txt
.\.venv\Scripts\python.exe tools/train_rl.py --smoke --hours 0.2 --workers 30 --output runs/train/smoke  # ~12 min
.\.venv\Scripts\python.exe tools/train_rl.py --hours 48 --workers 30
```

`penalty/rl/training.py` ejecuta el currículo de `docs/rl/06`: primero todas las etapas sintéticas (≈7 000 pasos/s, ~25 min en total, pasos fijos), luego B-Human, que se reparte el resto de horas por igual entre tirador (`static` 40 %, `scripted` 60 %) y portero (`random_kicks`). Cada rol conserva una política PPO (MLP 128×128, normalización de observaciones, `target_kl=0.03`) que pasa de etapa en etapa. El portero usa historia de 6 observaciones.

**Recolección asíncrona (etapas B-Human).** `SubprocVecEnv` avanza todos los entornos sincronizados y cada paso espera al más lento; tras una patada el entorno simula la pelota hasta el resultado (~440 ticks frente a ~9 de un paso normal) y el reinicio lanza dos motores y estabiliza 2 s. Con 16 entornos eso dejaba el CPU al 8 % y el entrenamiento en ~10 pasos/s. Ahora cada worker de `penalty/rl/actors.py` corre su entorno con una copia numpy de la política (sin torch) y envía transiciones sin esperar a los demás; `penalty/rl/async_ppo.py` llena el buffer con 4096 transiciones y entrena con la actualización PPO de stable-baselines3 sin cambios. Sigue siendo on-policy: los workers se pausan durante la actualización y retoman sus episodios. Entre segmentos de workers distintos se corta el GAE y se hace bootstrap con V(s') del mismo modo que SB3 al final de un rollout. Las pruebas comprueban que la política numpy coincide con la de torch, que las log-prob y valores guardados coinciden con los que recalcula el learner y que aprende igual que PPO (portero estático sintético: aleatoria 6.5 %, PPO 0.44–0.97, asíncrono 0.83–0.90 con 20 000 pasos y 3 semillas).

**Simulación por tick.** Optimizaciones sin cambio de resultados, comprobadas bit a bit (observaciones, recompensas, `qpos`/`qvel` finales) en 10 episodios con semilla de tirador y portero: lectura con búfer de los pipes nativos (antes `readline` hacía una lectura por byte), STEP a los dos motores antes de leer sus respuestas, contactos vectorizados, constantes de FSR y actuadores precalculadas, y sin trazas ni snapshots por tick cuando nadie los graba ni los lee. Tick de 8.0 a 2.3 ms y episodio de 7.6 a 2.2 s en un proceso aislado.

**Condiciones de entrenamiento (desde `ppo_v2`).** La corrida `ppo_v1_20261009` mostró dos atajos. El portero aprendió a agacharse en la guardia central antes de cada tiro: 0.778 de atajadas contra 0.420 del geométrico con las patadas uniformes. El tirador aprendió un solo tiro, en un mundo determinista. Por eso:

- Todas las etapas usan ruido de percepción; las B-Human añaden ±5 cm de balón y ±5 cm/±0.1 rad de pose inicial. El tirador v1 bajó de 1.00 a 0.81 de goles con esa variación; el portero v1 cayó a 0.09 de atajadas con 85 % de caídas.
- El portero entrena contra patadas a las esquinas: 3/4 apuntan a |y| 0.3–0.65 m, 1/4 al centro, con alcance de 1960–2900 mm. Aun así, la guardia central inmediata ataja 0.48 y el geométrico 0.28 (100 episodios): las patadas B-Human son imprecisas (21 % fuera) y la estirada lateral rinde poco. Por eso el reporte del portero añade el baseline `center` y la compuerta `primary_ge_center`. Superarla muestra que lee el tiro.
- Filtro de seguridad del portero (`commit_after_shot`, sintético y B-Human): una guardia o estirada pedida antes de que el rival toque el balón se ignora (`info['safety'] = 'waiting_for_shot'`). Así no tira el robot al suelo sin tiro y no puede comprometerse adivinando. En `ppo_v2` el portero sintético había aprendido tiros a los 0.5–3 s; en B-Human el rival patea a los 3.1–7.4 s (p10–p90, mediana 5.7 s), así que se lanzaba en la primera decisión y caía antes del tiro (0.13 de atajadas, 79 % `keeper_fall`). Con episodios de ~3 decisiones, la etapa solo hizo 27 actualizaciones PPO en una hora. Con tiros a los 3–9 s y sin filtro, PPO tampoco aprendió a esperar 30–90 decisiones (0.13–0.20 tras 3 M pasos sintéticos). Con el filtro: 0.636 en 1.5 M pasos (geométrico 0.672, guardia central 0.366).
- En B-Human no hay bono de entropía y la desviación de la Gaussiana se mantiene entre 0.2 y 0.5 antes de cada rollout. Con las acciones recortadas a [-1, 1], el bono la había inflado (portero 0.7–0.8; gatillo del tirador 20) y los episodios de entrenamiento eran casi aleatorios: 0.26 de atajadas al entrenar contra 0.79 sin ruido.

**Convergencia (etapas B-Human).** Cada worker juega 1 de cada 5 episodios con la acción media, como se usaría la política. Esos episodios no entran al rollout de PPO y forman ventanas de 200 para la convergencia y el mejor checkpoint. Sin `eval_every`, se usan los episodios de entrenamiento en ventanas de 1000. Una ventana mejora si su tasa principal (`goal_rate` o `save_rate`) supera a la mejor por más del error estándar de la diferencia entre dos ventanas (≈5 puntos con 200 episodios y p=0.5). La etapa converge con 4 ventanas seguidas sin mejora y al menos 60 min de etapa. El tiempo que libera pasa a las etapas pendientes, y `--hours` queda como tope. Al cerrar, si la última ventana es claramente peor que la mejor, se restaura `<rol>/<etapa>_best`. Que se estanque no garantiza el óptimo: PPO puede volver a mejorar después de una meseta.

**Protecciones.**

| Condición | Acción |
| --- | --- |
| `sim_error` > 10 % en una ventana | Aborta la corrida (código 3, sin reinicio): el runtime está mal. |
| Pesos no finitos (NaN) | No sobrescribe el último checkpoint válido, cierra la etapa como `diverged`, restaura el mejor y continúa. |
| 15 min sin pasos de entorno | El watchdog termina el proceso (código 75); el supervisor reanuda. |
| Caída o cuelgue del proceso | El supervisor de `tools/train_rl.py` reanuda con `--resume` tras 30 s; se rinde con más de `--max-restarts` (3) fallos seguidos de menos de 30 min. |
| Reinicio de B-Human fallido | 5 reintentos con sesión nueva antes de fallar. |
| Compuertas al cerrar etapa | `report.json` registra `primary_ge_baseline` y `falls_le_baseline_plus_5pct` frente al baseline; solo informan, no detienen. |

- Salida en `runs/train/<fecha>/`: `progress.json`/`report.json` con procedencia, reanudaciones (`resumes`), motivo de parada (`converged`, `deadline`, `diverged`, `budget`) y ventanas por etapa; `<rol>/latest`, `<rol>/<etapa>` y `<rol>/<etapa>_best` (`model.zip`, `obs_norm.npz`, `config.json`); registros CSV en `logs/<etapa>/` con `outcome/*_rate` y `convergence/*` (una etapa reanudada escribe en `logs/<etapa>/resume<k>/`).
- Tras cada etapa se compara contra el baseline (tirador scripted, portero geométrico) en semillas apartadas.
- Puntos de control cada 10 min y al interrumpir con Ctrl+C. `--output <carpeta> --resume` continúa: las etapas sintéticas incompletas se repiten, las de B-Human conservan el estado de convergencia y usan el tiempo restante.
- Windows no entra en suspensión mientras entrena.
- Evaluar un modelo en las condiciones de una etapa: `tools/evaluate_rl.py --role keeper --stage keeper_bhuman_random_kicks --policy runs/train/<fecha>/keeper/latest --episodes 100 --seeds 20000 20001 20002 20003 20004 --workers 30` (`--policy center` o `geometric` para los baselines).
- Duelo entre políticas (cross-play en B-Human, `penalty/rl/duel.py`): `tools/duel_rl.py --strikers scripted v2=<dir> --keepers scripted v4=<dir>:gate --episodes 200 --workers 30 --conditions realista sin_ruido`. Los dos controladores tácticos deciden cada 0.1 s con su propia observación y ruido; tras la patada el episodio no se adelanta, porque el portero sigue decidiendo hasta comprometerse. Todas las parejas juegan las mismas semillas (comparación pareada). `:gate` activa `commit_after_shot` para porteros entrenados con él. La inferencia de los workers es numpy (sin torch).
- Corridas, resultados y ubicación de modelos y reportes: [RESULTADOS_RL.md](RESULTADOS_RL.md).

Rendimiento medido el 2026-10-09 (Ryzen 9 8940HX, 16 núcleos/32 hilos, 31 GB, enchufado, modo de energía "Máximo rendimiento"):

| Recolección B-Human | Tirador vs `scripted` | CPU |
| --- | --- | --- |
| Sincronizada, 16 entornos (antes) | ~10 transiciones/s | 8 % |
| Asíncrona, 24 workers | 128 transiciones/s, 4.9 ep/s | — |
| Asíncrona, 30 workers | 148 transiciones/s, 5.0 ep/s | 100 % a ~3.2 GHz |

Portero `random_kicks` con 30 workers y política inicial aleatoria: 9.6 ep/s (se lanza enseguida; la entrenada espera el tiro y da más decisiones por episodio). Con 30 workers el CPU queda saturado: el límite es la potencia sostenida del portátil. Cada worker usa ~0.25 GB (30 workers ≈ 8 GB).

**GPU.** No acelera este entrenamiento. La actualización PPO de 4096 transiciones (MLP 128×128, 10 épocas, lote 256) tarda 0.3 s en CPU frente a ~28 s de recolección: ~1 % del tiempo, y con una red tan pequeña la GPU no sería más rápida por el costo de transferencia y lanzamiento de kernels. El costo está en la simulación (MuJoCo y el motor B-Human en C++ por robot), que corre en CPU; los simuladores en GPU (MJX, MuJoCo Warp) no pueden alojar el MotionEngine original por tick, que es el controlador de bajo nivel exigido.

Hallazgo de la prueba corta: la patada B-Human en MuJoCo está cerca de un límite de estabilidad. Con la misma semilla, liberar a 2248 mm a los 5.2 s derriba al tirador, mientras 2242 mm no lo hace; las peticiones al motor fueron idénticas byte a byte. Es determinista y se penaliza como caída, así que la política aprende a evitarlo. Hay que vigilar `any_fall_rate` frente al baseline.

## Estado frente al plan de 20 días

Hecho (P0, Agente A completo y partes no entrenables de B/C/E):

- Contratos, observación relativa con velocidad e historia, entornos con seeding, terminaciones, `info` y pruebas (`check_env`).
- Recompensas y objetivos versionados, métricas primarias/secundarias, baselines scripted y geométrica.
- Skills con límites duros y rechazo de valores inválidos; búsqueda aleatoria y CMA-ES.
- Evaluación multi-semilla/multiproceso con throughput y procedencia.
- Entrenamiento PPO (2026-10-08 a 10, [RESULTADOS_RL.md](RESULTADOS_RL.md)): recolección asíncrona, convergencia con episodios sin ruido, protecciones, condiciones robustas y duelo entre políticas. El tirador v2 pasa las compuertas (0.96 frente a 0.02 del scripted contra el portero FSM, con variación y ruido). El portero v4 supera al geométrico (0.37 frente a 0.18), pero no a la guardia central tras la patada (0.55).

Pendiente:

- Un portero entrenado que supere a la guardia central.
- Exponer parámetros de patada/bloqueo en el runtime nativo generado (días 4–6).
- Self-play y `opponent_pool.py` (P1). La matriz cross-play ya se puede medir con `tools/duel_rl.py`.
- Residual RL (P1), `penalty/rl/runtime.py` y adaptador de observación para NAO (Agente F).
- `--striker-controller/--keeper-controller rl` en el simulador cuando exista un formato de modelo.

Ningún resultado de esta capa afirma equivalencia con hardware.

## Portero con visión — 2026-10-10

`vision_keeper_v2` consume `rl_obs_v2`, con máscaras de medidas desconocidas,
captura y covarianza de velocidad. El portero recibe el balón desde píxeles,
proyección calibrada y seguimiento con odometría; no usa postura del rival.
Los contratos/checkpoints v1 se preservan. En 160 duelos de cámara simulada,
ocho estilos y semillas nuevas, hizo 74/80 atajadas frente a 44/80 del base.
Es preliminar y usa sensores/calibración ideales de simulación.

Comando: `tools/evaluate_keeper_vision.py --episodes 10 --workers 4 --seed 65001 --output runs/keeper_vision/camera.json`.
El modo NAOqi nuevo solo registra decisiones; falta validar en el robot y
calibrar el ejecutor físico. Diseño, alcance y protocolo de modelos B-Human:
[VISION_PORTERO.md](VISION_PORTERO.md).
