# Tirador preserving_v4 contra nuestro portero v2 balanced

El candidato no cumple todos los criterios predeclarados de mejora. Se conserva striker_v2 como tirador recomendado y este modelo queda experimental. El checkpoint de la actualización **64** se eligió con validación antes de ejecutar **600 penales reservados**. La prueba sólo decide la recomendación: no eligió ni modificó pesos.

## Prueba reservada: 100 penales por portero y tirador

| Portero | Tirador | Penales | Goles | Atajadas | Fuera/corto | Timeout | Caída terminal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nuestro portero v2 visual + balanced, movimiento original | v2 intacto | 100 | 12 (12,00 %) | 86 (86,00 %) | 2 | 0 | 0 |
| Nuestro portero v2 visual + balanced, movimiento original | preserving_v4 | 100 | 15 (15,00 %) | 83 (83,00 %) | 2 | 0 | 0 |
| Nuestro portero v2 visual + balanced + salto ajustado | v2 intacto | 100 | 19 (19,00 %) | 79 (79,00 %) | 2 | 0 | 0 |
| Nuestro portero v2 visual + balanced + salto ajustado | preserving_v4 | 100 | 23 (23,00 %) | 75 (75,00 %) | 2 | 0 | 0 |
| Nuestra FSM base con movimientos B-Human | v2 intacto | 100 | 93 (93,00 %) | 5 (5,00 %) | 2 | 0 | 0 |
| Nuestra FSM base con movimientos B-Human | preserving_v4 | 100 | 93 (93,00 %) | 5 (5,00 %) | 2 | 0 | 0 |

Cada pareja usa la misma semilla física y los mismos inicios dentro de su rival. Sólo `goal` cuenta como éxito; una atajada, fallo, caída terminal o timeout no cuenta como gol. `miss` es un tiro fuera/corto sin bloqueo. Una caída posterior a la patada se conserva en la traza aunque el balón termine en gol o atajada. Hubo cero `sim_error` en validación y prueba, y todos los hijos nativos fueron recogidos.

| Portero | Cambio de goles frente a v2 | IC95 pareado | Goles exclusivos nuevo / v2 |
| --- | ---: | --- | ---: |
| Nuestro portero v2 visual + balanced, movimiento original | +3,00 puntos | [-1,00; +8,00] puntos | 4 / 1 |
| Nuestro portero v2 visual + balanced + salto ajustado | +4,00 puntos | [-2,00; +10,00] puntos | 7 / 3 |
| Nuestra FSM base con movimientos B-Human | +0,00 puntos | [+0,00; +0,00] puntos | 0 / 0 |

Los intervalos usan 10.000 remuestreos de las parejas y todos los penales programados. La comprobación contra FSM exige una pérdida observada de como máximo 10 puntos; **no es una prueba formal de no inferioridad**. Los intervalos de rivales secundarios son descriptivos, sin corrección por comparaciones múltiples.

| Criterio de recomendación | Cumplido |
| --- | --- |
| Más goles que v2 en validación contra balanced | Sí |
| Pérdida observada contra FSM ≤10 puntos en validación | Sí |
| Límite inferior del IC95 pareado contra balanced >0 | No |
| Pérdida observada contra FSM ≤10 puntos en prueba | Sí |

## Qué se cambió al entrenar

Se completaron 96 actualizaciones PPO × 6 actores × 128 pasos: **73.728 transiciones nuevas optimizadas** y 902 episodios completos incorporados. Sus resultados de entrenamiento fueron 232 goles, 663 atajadas, 4 fallos, 3 timeout y 0 caídas terminales. El modelo seleccionado contiene 49152 transiciones nuevas, correspondientes a sus primeras 64 actualizaciones. Es una corrida acotada; no se afirma convergencia.

La nueva inicialización `striker_policy_init_v3` conserva **cada byte de los pesos del actor v2** y su normalización. No reescala la cabeza de acciones. En entrenamiento, si una media cruda supera +1 se sustituye por `1 + 0,01*(media-1)`; si está por debajo de −1, por `−1 + 0,01*(media+1)`. Dentro de `[-1,1]` no cambia. El recorte determinista sigue produciendo exactamente las acciones de v2. Las derivadas son las reales de esa función, sin gradiente sustituto.

Se reinicia Adam completo, se pone en cero la capa final del crítico y se inicia sigma en 0,25, acotada después a [0,12; 0,40]. La recompensa nueva `striker_visual_reward_v3` da cero por paso, +1 por gol y −1 por cualquier otro terminal físico. Con `gamma=1` y el horizonte observado de 28 segundos, su retorno esperado es `2*P(gol)-1`. Un error de simulación es diagnóstico fatal: el rollout se descarta antes de optimizarlo.

Cinco de los seis actores entrenan contra nuestro portero visual v2 balanced, con movimientos originales; uno juega contra nuestra FSM base para conservar esa habilidad. PPO usa learning rate 0,00003, clip 0,05, cuatro épocas por actualización, GAE 0,95, KL objetivo 0,01 y ningún bonus de entropía. El rival con salto ajustado participa sólo en prueba.

## Selección mediante validación

| Portero de validación | Política | Goles / 64 | Atajadas | Fallados | Timeout |
| --- | --- | ---: | ---: | ---: | ---: |
| Nuestro portero v2 visual + balanced, movimiento original | v2 | 13 / 64 | 50 | 1 | 0 |
| Nuestro portero v2 visual + balanced, movimiento original | initialized | 13 / 64 | 50 | 1 | 0 |
| Nuestro portero v2 visual + balanced, movimiento original | update_0032 | 13 / 64 | 50 | 1 | 0 |
| Nuestro portero v2 visual + balanced, movimiento original | update_0064 | 14 / 64 | 49 | 1 | 0 |
| Nuestro portero v2 visual + balanced, movimiento original | update_0096 | 12 / 64 | 51 | 1 | 0 |
| Nuestra FSM base con movimientos B-Human | v2 | 60 / 64 | 3 | 1 | 0 |
| Nuestra FSM base con movimientos B-Human | initialized | 60 / 64 | 3 | 1 | 0 |
| Nuestra FSM base con movimientos B-Human | update_0032 | 60 / 64 | 3 | 1 | 0 |
| Nuestra FSM base con movimientos B-Human | update_0064 | 60 / 64 | 3 | 1 | 0 |
| Nuestra FSM base con movimientos B-Human | update_0096 | 60 / 64 | 3 | 1 | 0 |

Se compararon las actualizaciones 32, 64 y 96 usando 64 parejas contra balanced y otras 64 contra FSM. Entre los checkpoints entrenados que cumplen el límite de pérdida de 10 puntos frente a FSM, se elige el que obtiene más goles contra balanced; desempata con menos caídas y después con la actualización más temprana. Si ninguno cumple, se exporta el mejor entrenado sólo para diagnóstico y v2 conserva la recomendación. La política inicializada es un diagnóstico, nunca un candidato entrenado.

## Fallos, semillas y reanudación

No se registró ninguna interrupción ni descarte de rollout en la corrida final. La evidencia completa incluye `failures.jsonl` si hubo incidencias, stdout/stderr, trazas y metadatos de semillas. Antes de cada rollout se guarda un intervalo conservador de posibles semillas, de forma que una interrupción no reutilice ninguna al reanudar. Los checkpoints incluyen Adam y RNG de Python/NumPy/Torch; los episodios físicos reinician con semillas nuevas, sin simular una restauración exacta de MuJoCo. Las 73.728 transiciones son las optimizadas: no se presentan como todo el trabajo físico de un rollout descartado.

Antes de la corrida, el primer helper de paridad falló al guardar una configuración porque repetía el campo `history`. Ese error de código ocurrió antes de ejecutar los duelos de comprobación, se corrigió en el helper y se conserva su log. La repetición completó cuatro duelos, con resultados y campos físicos idénticos entre v2 y la política inicializada. El smoke test posterior completó 64 transiciones, sin episodios terminados ni error; no se presenta como una evaluación de rendimiento. Estos ensayos están separados del entrenamiento final.

Entrenamiento, validación y prueba usan respectivamente bases 101001, 102001 y 103001; la semilla física es `base*1.000.000+índice`. La selección congelada impide seguir entrenando dentro de esta corrida.

## Percepción y compatibilidad

El portero recibe RGB renderizado y sensores propios simulados de encoders/IMU/FSR, con odometría estimada. Conserva v2 balanced y no recibe orientación ideal del rival. El tirador recibe todavía `rl_obs_v1` privilegiada de la escena con `PerceptionNoise v1`, igual que v2: no es un tirador guiado por cámaras reales. La FSM rival es nuestra implementación con motores B-Human, no el equipo oficial completo de B-Human.

Las cuatro acciones son objetivo lateral, pie, potencia y disparo. B-Human sigue ejecutando los movimientos; no se entrenaron trayectorias articulares. MuJoCo usa raíces libres, servos y contactos, con sensores propios ideales y pausa física durante inferencia RGB. No se ejecutó ni validó un NAO físico.

El exportador NumPy original sirve para **acciones deterministas recortadas**. Su muestreo gaussiano crudo y sus log-probabilidades no representan esta nueva compresión de medias; por eso este modelo no se reanuda con el entrenador asíncrono genérico. Se carga mediante la clase PPO guardada y el runner dedicado.

## Evidencia y reproducción

- [Prueba completa](test_report.json), [validación](validation_report.json), [selección congelada](selection.json), [inicialización](initialization_report.json) y [entrenamiento](training_report.json).
- [Manifiesto de corrida](run_manifest.json), [auditoría independiente final](final_audit.json), [paridad física de la inicialización](env_preservation.json), [367 pruebas aprobadas](full_tests.log), [resultados](results.json) y [SHA de todos los archivos](artifacts.json).
- [Modelo](../../../../models/rl/striker_preserving_v4/README.md) y [ZIP completo](complete_run.zip): 96 checkpoints con pesos, Adam y RNG, base v2, rival congelado, caches pareados, progreso, diagnósticos, logs y los 121 inputs originales de fuentes, modelos, mallas y runtime. `onnxruntime.dll` está incluido en el fingerprint original de esta corrida. Las pruebas adicionales del ZIP están etiquetadas como snapshot de soporte si no formaron parte de esos hashes.

Fingerprint: `3343ae9fee9a7ad2080bb26e9039fbc22958de62ebcadbaf60d322e9510256d7`.

SHA256 del ZIP: `ea5a820fc8c8f1247509c12cde3980534cda423258fa63bf3cbfc6096eb882ce`.

```powershell
# Restaurar inputs/ en sus rutas, run/ en runs/striker_retrain_v2/preserving_v4
# y support/ en runs/striker_retrain_v2. Verificar artifact_manifest.json.
python tools/retrain_striker_v2.py --output runs/striker_retrain_v2/preserving_v4 `
  --workers 6 --n-steps 128 --updates 96 --checkpoint-every 32 `
  --validation-episodes 64 --test-episodes 100 --learning-rate 0.00003 `
  --max-seconds 7200 --retention-max-drop 0.10 --stage test --resume
```

Este comando valida identidad y reutiliza la prueba cacheada. Una repetición física independiente requiere otra carpeta, inputs y versiones congeladas, y el protocolo completo.
