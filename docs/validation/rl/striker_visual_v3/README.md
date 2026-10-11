# Tirador v3 contra portero visual: piloto experimental con v2 balanced

Se afinó tácticamente `striker_v2` contra nuestro portero visual v2 con `balanced` y movimientos B-Human originales. El checkpoint de la actualización **16** quedó seleccionado por **32 tiros de validación**, antes de ejecutar la prueba reservada. El intervalo incluye cero: esta prueba no demuestra una ventaja de goles frente al portero principal. Se conserva striker_v2 como referencia. Los duelos contra Nuestro v2 visual + balanced + salto ajustado, Nuestra FSM base con movimientos B-Human muestran regresiones con intervalos completamente negativos. No se cambia la recomendación ni los archivos de `striker_v2` por el resultado de esta prueba.

## Prueba reservada: 100 tiros por rival y tirador

| Rival | Tirador | Penales | Goles | Atajadas | Fuera/corto | Timeout | Caída terminal |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nuestro v2 visual + balanced, movimiento original | striker_v2 intacto | 100 | 16 (16,00 %) | 84 (84,00 %) | 0 | 0 | 0 |
| Nuestro v2 visual + balanced, movimiento original | striker_visual_v3 experimental | 100 | 12 (12,00 %) | 87 (87,00 %) | 1 | 0 | 0 |
| Nuestro v2 visual + balanced + salto ajustado | striker_v2 intacto | 100 | 26 (26,00 %) | 74 (74,00 %) | 0 | 0 | 0 |
| Nuestro v2 visual + balanced + salto ajustado | striker_visual_v3 experimental | 100 | 15 (15,00 %) | 84 (84,00 %) | 1 | 0 | 0 |
| Nuestra FSM base con movimientos B-Human | striker_v2 intacto | 100 | 91 (91,00 %) | 9 (9,00 %) | 0 | 0 | 0 |
| Nuestra FSM base con movimientos B-Human | striker_visual_v3 experimental | 100 | 38 (38,00 %) | 61 (61,00 %) | 1 | 0 | 0 |

Cada pareja usa exactamente las mismas semillas y condiciones de inicio. Sólo `goal` es éxito: una atajada, un tiro fuera/corto, caída o timeout no cuentan como gol. “Fuera/corto” agrupa el resultado físico `miss`, sin contacto de bloqueo; “caída terminal” suma `striker_fall` y `keeper_fall`. Las caídas posteriores a una patada también están en las trazas, aunque el resultado del balón pueda ser gol o atajada. Hubo **cero errores de simulación** en esta evaluación y todos los hijos nativos fueron recogidos.

| Rival | Ganancia de goles | IC95 pareado | Goles exclusivos nuevo / v2 |
| --- | ---: | --- | ---: |
| Nuestro v2 visual + balanced, movimiento original | -4,00 puntos | [-12,00; +4,00] puntos | 7 / 11 |
| Nuestro v2 visual + balanced + salto ajustado | -11,00 puntos | [-21,00; -1,00] puntos | 8 / 19 |
| Nuestra FSM base con movimientos B-Human | -53,00 puntos | [-63,00; -43,00] puntos | 1 / 54 |

Los intervalos son bootstrap pareado del indicador de gol, con todos los episodios programados en el denominador. Cada intervalo corresponde a un rival; no se aplicó corrección por comparaciones múltiples. Este ensayo compara políticas en la misma simulación y no establece convergencia ni rendimiento general fuera de estos tres rivales.

## Qué se entrenó y qué aporta la inicialización

Se completaron **24 actualizaciones PPO** de 6 actores × 128 pasos: **18432 transiciones optimizadas en la corrida**. El checkpoint seleccionado de la actualización 16 contiene **12288 transiciones optimizadas nuevas**, además del entrenamiento heredado de v2. La política conserva `rl_obs_v1` (19 campos), `striker_action_v1` (4 decisiones: objetivo lateral, pie, potencia y disparo) y la normalización congelada de v2. No aprende trayectorias articulares: B-Human sigue ejecutando marcha y patada.

Antes de PPO se corrigió la saturación de la cabeza de acciones mediante `striker_policy_init_v2`, usando ocho tiros de calibración de semillas 90001. Las escalas por fila fueron `[0.1337672346745401, 0.02157656441062049, 0.3501386873106254, 0.021433359490509642]`; se conservaron los demás pesos y se reinició únicamente el estado Adam de esa cabeza. La calibración no consumió validación ni prueba. Por eso el progreso respecto a v2 debe distinguir **inicialización** de **aprendizaje PPO**.

| Política | Goles / 32 | Atajadas | Fallados | Timeout |
| --- | ---: | ---: | ---: | ---: |
| v2 | 7 / 32 | 25 | 0 | 0 |
| initialized | 2 / 32 | 30 | 0 | 0 |
| update_0008 | 0 / 32 | 32 | 0 | 0 |
| update_0016 | 4 / 32 | 28 | 0 | 0 |
| update_0024 | 4 / 32 | 28 | 0 | 0 |

La inicialización sola obtuvo 2/32 goles y no era elegible como candidato entrenado. El checkpoint elegido obtuvo 4/32: frente a la inicialización, su cambio fue +6,25 puntos, IC95 [-6,25; +18,75]. La selección examinó las actualizaciones 8, 16 y 24, favoreciendo más goles, después menos caídas y después la actualización más temprana. La prueba final no eligió ni ajustó pesos.

La prueba reservada compara el efecto completo de inicialización y entrenamiento con v2. No prueba la inicialización sola frente a los tres rivales; por tanto, una regresión en esos duelos no puede atribuirse únicamente a las actualizaciones PPO.

En los 100 casos reservados contra el rival principal, el candidato solicitó objetivos más centrales y menos potencia. El objetivo lateral absoluto medio pasó de 0,633 m en v2 a 0,285 m; los objetivos dentro de ±0,30 m pasaron de 2 a 53 tiros y los de al menos 0,60 m de 93 a 0. La potencia solicitada media pasó de 0,751 a 0,627. Esto describe las decisiones observadas, no identifica por separado una causa de la regresión ni equivale al punto final real del balón.

La comparación de promoción en validación favoreció a striker_v2 intacto; se prueba el mejor checkpoint entrenado como candidato experimental, sin reemplazarlo.

La recompensa nueva `striker_visual_reward_v2` cobra −0,02 por decisión, da +10 al gol y −10 a cualquier otro resultado terminal, incluido timeout. El límite observado de 28 segundos termina el episodio y no permite bootstrap hacia un futuro posterior al penal.

## Interrupción y recuperación

La primera sesión sufrió **un error de simulación durante la recolección**, después de la actualización 4. Su mensaje genérico no conservó la causa ni la semilla del episodio fallido; no se afirma que el entrenamiento haya tenido cero errores. El rollout incompleto no se optimizó. Se conservaron pesos, optimizador y RNG del último checkpoint completo, y se reservaron **774 índices de semillas** (70–843) para evitar reutilizar los posibles episodios de esa recolección descartada. La reanudación empezó desde 91001000844, sin cambiar las 116 fuentes/configuración congeladas.

El historial, los logs originales y de reanudación, el wrapper diagnóstico y la [auditoría de recuperación](recovery_audit.json) están preservados en el ZIP y en [failure_history.json](failure_history.json). Por esta interrupción, las 18432 transiciones optimizadas no son un conteo de todo el trabajo físico ejecutado: los episodios/transiciones de la recolección descartada no se conocen exactamente.

## Percepción y alcance

El portero principal recibe RGB renderizado, detección de balón y transformaciones derivadas de sus propios encoders/IMU/FSR y odometría estimada. Conserva la predicción v2 y la selección `balanced`; no usa una orientación ideal del rival. El tirador recibe datos privilegiados de la escena con `PerceptionNoise v1`, como v2: su percepción aún no equivale a cámaras reales. El rival secundario carga el perfil congelado `keeper_motion_v1` y acelera un 15 % las fases activas del salto. La FSM base es nuestra implementación con movimientos B-Human, **no el comportamiento oficial completo de B-Human**.

Todo ocurrió en MuJoCo con raíces libres, servos y contactos. Los sensores propios simulados son ideales y la física se pausa durante inferencia RGB. **No hubo ejecución ni validación en un NAO físico.**

## Evidencia y reproducción

- [Prueba completa](test_report.json), [selección congelada](selection.json), [validación](validation_report.json), [entrenamiento](training_report.json) y [manifiesto de corrida](run_manifest.json).
- [Inicialización](initialization_report.json), [auditoría de calibración](calibration_audit.json), [paridad del entorno en dos semillas externas](env_parity.json), [auditoría independiente](final_audit.json) y [337 pruebas aprobadas](full_tests.log).
- [Resumen de resultados](results.json), [modelo exportado](../../../../models/rl/striker_visual_v3/README.md) y [todos los SHA exportados](artifacts.json).
- [ZIP reproducible de toda la corrida](complete_run.zip): 24 checkpoints con optimizador y RNG, pesos originales, calibración, rival congelado, progreso, logs, evaluaciones y todos los 116 inputs congelados de fuentes/modelos/runtime/mallas. Incluye adicionalmente `onnxruntime.dll`, con SHA propio, porque esa dependencia no estaba en la clave original de 116 inputs.

Fingerprint de corrida: `dea1c26dc1fe338f884b86edccea6ef84e988dd65a0e3b7529c2e131b35fb7ed`.

SHA256 del ZIP: `6dfa4bd6873e00adfcdabd68520113b1c135e7e8fa696ebe5c88726338211b3e`.

```powershell
# En una copia del repositorio: restaurar inputs/ respetando sus rutas,
# run/ en runs/striker_retrain/visual_v3, y support/ en runs/striker_retrain.
# Verificar los SHA de artifact_manifest.json antes de usar los archivos.
python tools/retrain_striker.py --output runs/striker_retrain/visual_v3 `
  --calibration-report runs/striker_retrain/visual_v3/frozen/calibration_report.json `
  --workers 6 --n-steps 128 --updates 24 --checkpoint-every 8 `
  --validation-episodes 32 --test-episodes 100 --learning-rate 0.0001 `
  --max-seconds 3600 --stage test --resume
```

Ese comando reutiliza los episodios de prueba ya guardados y comprueba identidad; no selecciona ni entrena otra política. Una repetición física nueva requiere otra carpeta y reproducir el protocolo completo, con las versiones y los inputs indicados en el manifiesto. Los checkpoints incluyen pesos y estado del optimizador; `resume_state.json` conserva RNG de Python/NumPy/Torch. Una reanudación de entrenamiento arranca episodios físicos nuevos con semillas sin usar, no restaura un estado exacto de MuJoCo. La selección ya congelada impide continuar entrenando dentro de esta corrida.
