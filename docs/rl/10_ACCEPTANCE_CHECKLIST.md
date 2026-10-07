# Acceptance checklist

## Gate 0 — baseline
- [ ] suite existente pasa
- [ ] simulator headless funciona
- [ ] striker/keeper scripted funcionan
- [ ] logs y seeds reproducibles

## Gate 1 — observations
- [ ] ball_vx/vy
- [ ] opponent relative pose
- [ ] goal geometry
- [ ] history wrapper
- [ ] final policy sin global coordinates

## Gate 2 — parameterized skills
- [ ] kick/block params validados
- [ ] hard bounds
- [ ] invalid values rejected
- [ ] metrics logged

## Gate 3 — optimization
- [ ] multiple seeds
- [ ] best params saved
- [ ] no material fall-rate regression
- [ ] evaluation report

## Gate 4 — striker RL
- [ ] Gym API valid
- [ ] PPO sin NaN
- [ ] benchmark vs scripted
- [ ] model/config guardados

## Gate 5 — keeper RL
- [ ] synthetic curriculum
- [ ] B-Human evaluation
- [ ] save/reaction metrics
- [ ] model/config guardados

## Gate 6 — self-play
- [ ] opponent pool
- [ ] snapshots
- [ ] cross-play matrix
- [ ] no catastrophic regression

## Gate 7 — sim-to-real
- [ ] noise/latency
- [ ] perception-compatible observation
- [ ] runtime benchmark
- [ ] scripted fallback

## Gate 8 — release
- [ ] code/model/config frozen
- [ ] hashes saved
- [ ] backup policy
- [ ] ABORT verified
- [ ] startup + rollback documented
