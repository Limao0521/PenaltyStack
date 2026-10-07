# Arquitectura y contratos

## Capas

```text
L3 Tactical Policy     -> qué hacer
L2 Skill Policy        -> cómo ejecutar la acción
L1 B-Human             -> balance, motion engines y actuadores
```

La política táctica no controla directamente los 23 servos.

## Observación canónica

Crear `penalty/rl/contracts.py`:

```python
@dataclass
class RLObservation:
    time_s: float
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    ball_age: float
    opponent_x: float
    opponent_y: float
    opponent_yaw: float
    goal_center_bearing: float
    left_post_bearing: float
    right_post_bearing: float
    torso_roll: float
    torso_pitch: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    remaining_time: float
```

Todas las posiciones deben convertirse a frame relativo al robot. No entrenar una policy final con coordenadas globales perfectas.

## Historia temporal

Keeper: usar `ObservationHistory(length=6)` y frame stacking antes de considerar LSTM.

## Acción striker

```python
@dataclass
class StrikerAction:
    target_y: float
    kick_side: int
    kick_power: float
    trigger: float
```

## Acción keeper

```python
@dataclass
class KeeperAction:
    block_side: int
    trigger: float
    amplitude: float
```

## Skills

```python
KickParameters(power, swing_amplitude, hip_rotation,
               foot_angle, contact_phase, support_lean)
```

```python
BlockParameters(lateral_amplitude, torso_lean, knee_flexion,
                arm_extension, activation_speed)
```

## Compatibilidad

Mantener controller scripted:

```text
--striker-controller scripted|rl
--keeper-controller scripted|rl
```

## Estructura sugerida

```text
penalty/
  rl/
    contracts.py
    observation.py
    wrappers.py
    striker_env.py
    keeper_env.py
    selfplay_env.py
    rewards.py
    metrics.py
    policies/
  skills/
    kick.py
    block.py
    residual.py
  optimization/
    kick_search.py
    block_search.py
    objective.py
```

Toda ejecución debe registrar seed, commit, versiones de reward/observation/action, opponent, modelo y resultado.
