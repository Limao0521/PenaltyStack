# Reentrenamiento del tirador contra el portero visual

Este experimento continúa `models/rl/striker_v2` contra nuestro portero visual
**v2 `balanced`, con movimientos B-Human originales**. El ajuste de movimiento
`keeper_motion_v1` se reserva como rival adicional: su mejora global anterior
no fue concluyente y no superó al movimiento original frente al tirador v2.

El portero recibe imágenes RGB de MuJoCo y sensores propios simulados
(encoders, IMU y FSR). El tirador mantiene `rl_obs_v1`: 19 campos de la escena
con ruido `PerceptionNoise v1`, iguales a su contrato publicado. Sus cuatro
acciones eligen dirección, pie, potencia y activación; B-Human ejecuta marcha
y patada mediante servos y contactos físicos. No se aprenden trayectorias
articulares ni se escribe la raíz del robot o la velocidad del balón.

## Inicialización

Las medias gaussianas de v2 estaban saturadas al recortarlas a `[-1, 1]`.
Ocho penales de calibración, fuera de entrenamiento, validación y prueba,
determinan una escala por salida: `min(1, 0.75 / max(p95(abs(media)), 0.75))`.
Solo se escalan las cuatro filas y sesgos de la cabeza de acciones; se
conservan las redes internas y de valor, y se eliminan únicamente los
momentos Adam incompatibles de esa cabeza. Es una inicialización parcial
desde v2, versionada `striker_policy_init_v2`, no una copia idéntica de v2.

La normalización de observaciones se conserva congelada, sin normalizar
recompensas. Una evaluación separada de la inicialización permite distinguir
su efecto del aprendizaje PPO posterior. Esta inicialización sin entrenamiento
no puede seleccionarse como nuevo modelo entrenado.

## Protocolo fijado antes de entrenar

- Calibración: ocho semillas `90001000000` a `90001000007`.
- Entrenamiento: 24 actualizaciones PPO, 128 pasos por seis entornos:
  **18 432 transiciones nuevas** como presupuesto; límite de una hora,
  comprobado entre actualizaciones. La corrida no demuestra convergencia.
- Validación: 32 semillas nuevas `92001000000` a `92001000031`, compartidas
  por v2, la inicialización y las actualizaciones 8, 16 y 24. Selección entre
  modelos entrenados por más goles, menos caídas y actualización más temprana.
  Se compara por separado si la validación prefiere v2 para conservarlo.
- Prueba independiente: 100 semillas `93001000000` a `93001000099` por rival,
  emparejadas entre v2 y el modelo entrenado seleccionado: **600 penales**.
  Rivales: visual `balanced`, visual `balanced` con `keeper_motion_v1`, y FSM
  base del proyecto. El mismo número de semilla no implica la misma escena
  entre los distintos tipos de rival; el emparejamiento se comprueba dentro
  de cada rival, entre ambos tiradores.
- Las semillas de entrenamiento pertenecen al espacio `91001 * 1000000 + i`;
  cada entorno recibe índices distintos. PPO explora con semilla 91001.
- Se congelan versiones, fuentes, modelos, calibración, rivales y sus hashes.
  Ningún resultado de prueba selecciona o reajusta el modelo de esta corrida.

La recompensa nueva `striker_visual_reward_v2` paga +10 por gol y -10 por
cualquier otro final, además de -0.02 por decisión. El límite observado de
28 s es terminal: un timeout no recibe recompensa preferible a una atajada.
Se conserva intacta la recompensa publicada v1. PPO usa tasa `1e-4`, clip
`0.15`, diez épocas, gamma `0.99`, GAE `0.95`, KL objetivo `0.03` y desviación
gaussiana limitada a `[0.15, 0.45]` entre recolecciones y checkpoints.

Las métricas cuentan goles, atajadas, fuera, caídas, timeout y errores como
resultados distintos. El denominador incluye todos los penales programados;
no se contabiliza un tiro fuera como atajada. La ganancia de goles tiene
intervalo del 95 % mediante bootstrap de diferencias emparejadas.

## Ejecución y archivos

```powershell
.venv/Scripts/python.exe tools/calibrate_striker_retraining.py --output runs/striker_retrain/calibration_v1 --episodes 8 --workers 4
.venv/Scripts/python.exe tools/retrain_striker.py --output runs/striker_retrain/visual_v3 --calibration-report runs/striker_retrain/calibration_v1/calibration_report.json
```

El segundo comando congela, entrena, valida y prueba. `--stage` permite
ejecutar esas etapas por separado; `--resume` exige las mismas opciones y
fuentes. La reanudación restaura PPO, optimizador y generadores aleatorios,
pero empieza episodios físicos nuevos con semillas aún no utilizadas; no
restaura exactamente el estado interno del simulador.

La corrida guarda `run_manifest.json`, `initialization_report.json`,
`progress.json`, checkpoints inmutables con estado de reanudación,
`training_report.json`, `selection.json`, evaluaciones por penal y
`test_report.json`. Las filas de evaluación verifican el cierre de todos
sus procesos nativos. Ante un error de simulación se detiene la selección.

## Alcance

Es simulación MuJoCo con control motor B-Human. Los sensores propios son
ideales, la inferencia RGB pausa el tiempo físico y el tirador sigue usando
información privilegiada de escena. La FSM base es la del proyecto con
movimientos B-Human: no es el comportamiento oficial completo de B-Human.
Esta corrida no valida cámaras, movimientos ni rendimiento del NAO físico.

Los resultados y la decisión de uso se documentan en el reporte versionado
`docs/validation/rl/striker_visual_v3/`; los experimentos anteriores se
conservan sin alterar sus cifras.
