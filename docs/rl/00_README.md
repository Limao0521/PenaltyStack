# PenaltyStack RL — Implementation Pack

Este paquete especifica la evolución de PenaltyStack hacia una arquitectura jerárquica para penales NAO V6.

## Arquitectura objetivo

```text
PERCEPTION
   │
   ├── STRIKER Tactical RL ──> target / foot / timing
   │            │
   │       Skill optimizer / Residual RL
   │            │
   └────────> B-Human Motion Engine
   │
   └── KEEPER Tactical RL ──> side / timing / amplitude
                │
           Skill optimizer / Residual RL
                │
           B-Human Motion Engine
```

Principio: **B-Human permanece como low-level controller.** RL decide y optimiza acciones sin reaprender locomoción completa salvo que un experimento demuestre una mejora clara.

Orden: instrumentación → skills parametrizadas → optimización → Gym envs → PPO striker → PPO keeper → domain randomization → opponent pools/self-play → sim-to-real → freeze de competencia.

Documentos:
- `01_ARCHITECTURE_AND_CONTRACTS.md`
- `02_STRIKER.md`
- `03_GOALKEEPER.md`
- `04_SKILL_OPTIMIZATION_AND_RESIDUAL_RL.md`
- `05_ENVIRONMENTS_REWARDS_AND_METRICS.md`
- `06_SELF_PLAY_AND_CURRICULUM.md`
- `07_SIM_TO_REAL.md`
- `08_20_DAY_PLAN.md`
- `09_AGENT_TASKS.md`
- `10_ACCEPTANCE_CHECKLIST.md`

## Prioridades

P0: mantener baseline, observaciones correctas, skills parametrizadas, métricas, entornos reproducibles y una política táctica funcional por rol.

P1: residual RL, domain randomization, opponent pools y self-play.

P2: full-joint RL, MARL conjunto o aprendizaje end-to-end desde RGB.

Desde el día 18 no se cambia arquitectura, reward, observation space ni action space.
