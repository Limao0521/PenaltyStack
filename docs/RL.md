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
| `penalty/rl/rewards.py`, `metrics.py` | `striker_reward_v1`, `keeper_reward_v1`, tasas con IC95 de Wilson. |
| `penalty/rl/policies/baselines.py` | Baselines scripted (equivalentes a la FSM), geométrica de portero y aleatoria. |
| `penalty/rl/evaluation.py`, `tools/evaluate_rl.py` | Benchmark multi-semilla, multiproceso, throughput e informe con procedencia. |
| `penalty/rl/run_metadata.py` | Commit, estado sucio, versiones, hashes y plataforma de cada ejecución. |
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
- Portero: oponentes `scripted` (tirador FSM, centro, pie izquierdo, 2500 mm) y `random_kicks` (patadas B-Human con objetivo, pie y alcance aleatorios por episodio).
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

## Estado frente al plan de 20 días

Hecho (P0, Agente A completo y partes no entrenables de B/C/E):

- Contratos, observación relativa con velocidad e historia, entornos con seeding, terminaciones, `info` y pruebas (`check_env`).
- Recompensas y objetivos versionados, métricas primarias/secundarias, baselines scripted y geométrica.
- Skills con límites duros y rechazo de valores inválidos; búsqueda aleatoria y CMA-ES.
- Evaluación multi-semilla/multiproceso con throughput y procedencia.

Pendiente:

- **Entrenamiento PPO** (días 10–15): requiere `stable-baselines3`/`torch` y horas de simulación. B-Human corre a ~1× tiempo real, ~7–10 s por episodio y proceso.
- Exponer parámetros de patada/bloqueo en el runtime nativo generado (días 4–6).
- Self-play, `opponent_pool.py`, matriz cross-play (P1).
- Residual RL (P1), `penalty/rl/runtime.py` y adaptador de observación para NAO (Agente F).
- `--striker-controller/--keeper-controller rl` en el simulador cuando exista un formato de modelo.

Ningún resultado de esta capa afirma equivalencia con hardware.
