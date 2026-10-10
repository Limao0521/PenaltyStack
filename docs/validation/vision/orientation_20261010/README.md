# Validación de orientación visual del portero

Fecha 2026-10-10. Procedimiento y comandos en
[ORIENTACION_PORTERO.md](../../../ORIENTACION_PORTERO.md).

- `probe.json` y `angular_trace.jsonl`: 12 giros iniciales reservados,
  144 capturas, 108 orientaciones válidas; error mediano 1.14°, p95 3.41°.
- `simulation_smoke.json`: 24 duelos pareados, ocho estilos, v3/v2/FSM;
  1533/1989 capturas con yaw válido, mediana 1.66°, p95 5.12°.
  V3 y v2 5/8 atajadas, FSM 2/8; cero errores/compromisos tempranos.
- `episode.json`, `episode_trace.jsonl` y `camera.png`: penal contra tirador
  entrenado v2, semilla 68003. Reproducción exacta de 71/71 decisiones.
- `camera_*.png`: ejemplos RGB originales de giros −177°, −87° y +3°.
- `validation.json`: 183 pruebas, hashes de fuentes/artefactos/modelo,
  integridad de originales y límites de los sensores.

Las poses verdaderas se consultan en `evaluation_only`, fuera de la entrada
de percepción/política. La percepción usa píxeles y sensordata propios, con
actitud/cámaras/odometría estimadas. El controlador de bajo nivel conserva
su perfil `mujoco-truth-v2` con orientación propia ideal. Sensores ideales,
hipótesis de apoyo/geometría, apariencia simulada y reloj de cómputo sin
avance físico: no es equivalencia con hardware. No hubo movimientos NAO.

La muestra de duelos es pequeña y no demuestra una ganancia adicional de
atajadas por orientación. Los resultados históricos permanecen separados.
