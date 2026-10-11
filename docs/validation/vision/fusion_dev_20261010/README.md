# Desarrollo de v4/v5 (semillas 69001)

Fecha 2026-10-10/11. Procedimiento y conclusiones en
[PORTERO_FUSION.md](../../../PORTERO_FUSION.md). Estas corridas usaron una
compilación Linux propia de los mismos hosts (movimiento B-Human y detector)
en un entorno de 2 núcleos, sólo para iterar; no sustituyen el test final en
Windows.

- `observer_69001.json`: 200 penales con un portero que nunca se compromete
  (25 por estilo). Las trazas por cuadro (9 MB) se regeneran con
  `tools/evaluate_keeper_fusion.py --keepers observer --episodes 25 --seed 69001 --frames`.
- `heading_fit_69001.json`: ajuste de `kick_heading_v1`. Incluye filas por tiro
  con el cruce libre (sólo etiqueta) y la postura congelada.
- `boundary_c.json`, `boundary_c_truth.json`: 64 semillas con |cruce| 0.18–0.36 m
  y su cruce libre.
- `probe_boundary.json`: esas semillas con tres acciones de v5 sin orientación:
  estirada al lado (`corner_y=0.15`), centro inmediato (`corner_y=0.45`,
  `centre_confidence=0`) y centro a 1.8 s.
- `probe_center.json`: estilo `center`, v2 frente a v5 con centro tardío.

La verdad del simulador sólo se usa para etiquetas y desgloses; ninguna
política la recibe.
