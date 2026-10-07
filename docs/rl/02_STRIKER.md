# Striker

## Objetivo

Separar decisión táctica y optimización motriz.

Mantener inicialmente:

```text
SEARCH_BALL -> APPROACH_BALL -> ALIGN_TO_BALL -> SELECT_SHOT -> KICK -> RECOVER
```

RL sustituye primero la decisión `SELECT_SHOT`, no locomoción.

## Observación

Mínimo:

```text
ball relative pose
goal geometry
keeper relative pose
keeper motion estimate
remaining time
robot stability
```

## Acción

```text
target_y
kick_side
kick_power
trigger
```

## Safety filter

No permitir kick si el balón está fuera de rango/bearing permitido, B-Human está busy o el robot está inestable.

## Reward táctica v1

```text
+10 goal
-8 blocked
-6 wide/out
-10 fall
-0.02/timestep
```

Shaping secundario: precisión de cruce, velocidad útil y tiempo. Métrica primaria: `goal_rate`.

## Skill optimization

Primero optimizar:

```text
power
swing_amplitude
hip_rotation
foot_angle
support_lean
contact_phase
```

Orden: bounded random search -> CMA-ES -> opcional Bayesian optimization.

Registrar por patada: parámetros, pose inicial, contacto, velocidad de salida, crossing_y, resultado, caída y duración.

## Residual RL

Después de tener una patada estable:

```text
q_final(t) = q_BHuman(t) + clip(delta_q_RL(t))
```

Abrir solo 6–10 acciones relevantes, comenzando por HipPitch/HipRoll/KneePitch/AnklePitch/AnkleRoll.

## Acceptance

En >=100 episodios/configuración y varias seeds:

```text
goal_rate >= baseline
fall_rate <= baseline + 5%
wide_rate < baseline
```
