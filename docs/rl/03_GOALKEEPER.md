# Goalkeeper

## Objetivo

Aprender trayectoria, punto de llegada, lado, timing y ejecución del bloqueo.

## Observación

```text
ball_x, ball_y
ball_vx, ball_vy
ball history N=6
striker relative pose/yaw
keeper torso state
remaining time
```

Implementar además un baseline geométrico:

```python
predict_goal_crossing(ball_position, ball_velocity)
```

con `time_to_goal` y `predicted_goal_y`.

## Acción

```text
WAIT
BLOCK_LEFT
BLOCK_CENTER
BLOCK_RIGHT
```

más `trigger` y `amplitude`.

## Reward v1

```text
+10 save
+4 meaningful contact
-10 goal
-5 fall without save
-0.02/timestep
```

Agregar early-dive penalty. Una caída con save no se penaliza igual que una caída inútil.

## Skill optimization

Exponer:

```text
lateral_amplitude
torso_lean
knee_flexion
arm_extension
activation_speed
```

Optimizar save probability, coverage y reaction time.

## Curriculum

```text
synthetic trajectories
-> parameterized B-Human kicks
-> frozen striker
-> opponent pool
-> self-play
```

## Métricas

`save_rate`, `reaction_time`, `wrong_side_rate`, `early_dive_rate`, `contact_rate`, `fall_rate`.

Acceptance: save rate superior al scripted baseline en tiros center/left/right y diferentes velocidades.
