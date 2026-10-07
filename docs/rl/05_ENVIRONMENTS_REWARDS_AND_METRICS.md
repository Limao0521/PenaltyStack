# Environments, rewards y métricas

Crear `PenaltyStrikerEnv`, `PenaltyKeeperEnv` y luego `PenaltySelfPlayEnv`, compatibles con Gymnasium.

Cada entorno implementa `reset(seed=None, options=None)`, `step(action)`, `close()`, `observation_space`, `action_space`.

## Termination striker

`goal`, `blocked`, `wide`, `out`, `fall`, `timeout`.

## Termination keeper

`save`, `goal`, `out`, `keeper_fall`, `timeout`.

## Reward versioning

Nunca editar una reward sin cambiar identificador: `striker_reward_v1`, `keeper_reward_v1`, etc.

## Métricas striker

Primarias: goal/blocked/wide/fall rates.
Secundarias: ball speed, target error, kick time.

## Métricas keeper

Primarias: save/goal rate.
Secundarias: reaction time, wrong side, early dive, contact, fall.

## Evaluación

Mínimo recomendado:
- >=5 seeds.
- >=100 episodios por configuración importante.
- Siempre contra scripted baseline.

Medir también throughput con 1, 4, 8 y 12 entornos para determinar el paralelismo real que permite el Ryzen/B-Human/MuJoCo.
