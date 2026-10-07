# Plan de 20 días

## Días 1–3
Física y baseline: arreglar patada derecha, caracterizar ambos kicks y blocks, exponer ball_vx/vy y opponent pose, generar `baseline_report.json`.

## Días 4–6
Implementar `ParameterizedKick` y `ParameterizedBlock`; random search + CMA-ES. Guardar `best_kick_params.json` y `best_block_params.json`.

## Días 7–9
Implementar Gym envs, seeding, rewards, terminations y tests.

## Días 10–12
PPO striker: empty goal -> static keeper -> scripted keeper. Gate: superar striker scripted baseline.

## Días 13–15
PPO keeper: synthetic trajectories -> B-Human -> optimized kicks. Gate: superar keeper scripted baseline.

## Día 16
Domain randomization y fine-tuning.

## Día 17
Opponent pool y self-play limitado.

## Día 18 — FREEZE
No cambiar arquitectura, reward, observations ni action spaces.

## Día 19
Hardware: latency, perception, ejecución, caídas, ABORT, recovery y sim-vs-real.

## Día 20
Competition release: commit/model/config hashes, primary policy, backup policy y scripted fallback.

Stop rule: una línea experimental que consume >1 día sin mejorar la métrica primaria se suspende.
