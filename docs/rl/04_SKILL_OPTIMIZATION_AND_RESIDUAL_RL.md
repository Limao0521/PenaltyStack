# Skill Optimization y Residual RL

## Orden de métodos

```text
1 parameter search
2 CMA-ES / evolutionary optimization
3 conditional skill
4 residual RL
5 full-joint RL solo si sobra tiempo
```

Para espacios de 4–8 parámetros, optimización directa debe preceder PPO.

## API

```python
class ParameterizedSkill:
    def reset(self): ...
    def apply(self, params): ...
    def bounds(self): ...
```

## Objective kick

```text
J = +goal + ball_speed - target_error - fall - duration
```

## Objective block

```text
J = +save + contact + coverage - useless_fall - reaction_time
```

Versionar pesos.

## Robustez

Evaluar cada candidato con múltiples seeds y variaciones de ball friction, ball pose, robot pose, servo strength, timing jitter y sensor noise.

## Residual RL

```text
base = B-Human
residual = RL
final = base + residual
```

Usar límites duros y un safe envelope pequeño. Terminar episodio por tilt severo, non-finite state o violación de límites.

Entrada residual: qpos/qvel, IMU, FSR, ball pose, phase progress y target.

Si residual RL no supera el mejor optimizer baseline dentro del presupuesto de competencia, se descarta.
