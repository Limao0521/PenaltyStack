# Self-play y curriculum

No empezar con self-play.

```text
stable skill
-> stable tactical policy
-> frozen opponent
-> opponent pool
-> self-play
```

## Opponent pool

Mantener snapshots de striker/keeper y baselines scripted/random.

Sampling inicial sugerido:

```text
40% current
30% historical strong
20% scripted
10% random
```

Una política no se promociona si solo mejora contra el oponente actual.

## Curriculum striker

```text
empty goal
static keeper
random keeper
scripted reactive keeper
frozen RL keeper
opponent pool
```

## Curriculum keeper

```text
synthetic trajectories
parameterized kicks
B-Human baseline striker
optimized striker
frozen RL striker
opponent pool
```

Construir una matriz de cross-play entre versiones para detectar ciclos.

No es necesario MAPPO inicialmente: usar PPO independiente y congelar el oponente durante ventanas de entrenamiento.
