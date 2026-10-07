# Tareas por agente

## Agent A — Contracts & Environments

Owner:
`penalty/rl/contracts.py`, `observation.py`, `wrappers.py`, `striker_env.py`, `keeper_env.py`.

Entrega: spaces, relative frames, history, seeding, termination, info y tests.

## Agent B — Striker

Owner:
`penalty/skills/kick*`, `penalty/rl/policies/striker*`, `optimization/kick*`.

Entrega: parameterized kick, optimizer, PPO striker, benchmark y checkpoint.

## Agent C — Keeper

Owner:
`penalty/skills/block*`, `penalty/rl/policies/keeper*`, `optimization/block*`.

Entrega: geometric predictor, parameterized block, optimizer, PPO keeper y benchmark.

## Agent D — Residual RL

Owner:
`penalty/skills/residual.py`, `penalty/rl/residual_env.py`.

Solo empieza cuando P0 esté estable. Debe demostrar que mejora o iguala optimizer baseline sin degradar seguridad.

## Agent E — Self-play & Evaluation

Owner:
`selfplay_env.py`, `opponent_pool.py`, `evaluation.py`, `tools/evaluate_rl.py`.

Entrega: pool, sampling, cross-play matrix, promotion rules y reportes.

## Agent F — Sim-to-real

Owner:
`deploy/`, `penalty/nao.py`, `penalty/rl/runtime.py`.

Entrega: hardware observation adapter, export/runtime benchmark, latency/noise validation y fallback.

## Orden de merge

```text
A
↓
B + C
↓
E
↓
D
↓
F
```

Branches sugeridas:

```text
feat/rl-contracts
feat/striker-rl
feat/keeper-rl
feat/skill-optimization
feat/residual-rl
feat/self-play
feat/rl-deploy
```

Cada PR debe incluir: what changed, why, tests, benchmark y known limitations.
