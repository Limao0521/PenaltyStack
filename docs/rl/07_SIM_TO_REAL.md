# Sim-to-real

Toda feature final debe tener un camino realista hacia hardware.

```text
MuJoCo truth -> robot-relative representation -> noise -> latency -> policy
```

No usar global exact world coordinates como dependencia final.

## Fases

A. privileged state
B. noisy state
C. perception-compatible state
D. replay de logs reales del NAO
E. hardware

## Domain randomization

Agregar progresivamente:
- friction/restitution del balón;
- pequeña variación de masa;
- servo strength y latency;
- poses iniciales;
- perception noise;
- observation delay/dropout.

Crear `DelayedObservationWrapper`.

Representar pérdida visual mediante `ball_age` y visibility masks.

Si una feature no puede observarse confiablemente en el NAO, eliminarla antes del día 16.

Mantener siempre un scripted fallback en hardware.
