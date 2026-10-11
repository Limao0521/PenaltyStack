# Test final de porteros v2–v5 (semillas 70001)

2026-10-11, Windows, ejecutables MSVC validados, 16 workers. Detalle y
conclusiones en [PORTERO_FUSION.md](../../../PORTERO_FUSION.md).

- `final_70001.json`: duelo pareado de seis porteros (FSM base, v2, v3, v4,
  v5 balón, v5). Cada uno juega 50 penales por estilo en ocho estilos, con los
  mismos tiros para todos.
- `final_70001.log`: progreso y resumen; `tests.log`: suite previa (198 OK).
- `observer_70001.json`: portero que nunca se compromete en las mismas semillas.
  Sólo sirve para analizar dónde cruzaría cada balón. Las trazas por cuadro
  (20 MB) quedan en `runs/keeper_fusion/` y se regeneran con
  `tools/evaluate_keeper_fusion.py --keepers observer --episodes 50 --seed 70001 --frames`.
- `analysis_70001.json`: tablas, comparaciones, diagnóstico de decisiones y
  atajadas por cruce real (`tools/analyze_keeper_fusion.py`).

Los `sim_error` son tiempos de espera del detector nativo y cuentan como no
atajadas. Las semillas 70001 no se usaron para elegir parámetros.
