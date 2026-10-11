# Selector de seis defensas sobre el portero visual v2

**Entrenamiento y prueba ciega completados.** El selector obtuvo **135/160 atajadas (84.375 %)**, v2 original **128/160 (80 %)** y `balanced` **134/160 (83.75 %)**. La ganancia aprendida frente a `balanced` es una atajada, con un intervalo que incluye empate: no demuestra una mejora fiable por aprender seis opciones. La regla del centro espejado captura casi toda la mejora observada.

Se entrena qué defensa original B-Human elegir cuando `vision_keeper_v2` decide actuar. Las seis opciones son `center`, `center_right`, `wide_left`, `wide_right`, `left` y `right`. La preparación `prepare` queda fuera del selector. No se entrenan ni modifican trayectorias articulares.

`vision_keeper_v2` es el portero de reglas temporales que utiliza el balón observado por cámara. Es distinto de `models/rl/keeper_v2`, el PPO anterior que se lanzaba antes del tiro y está marcado como «no usar». Esta corrida conserva el predictor visual v2 y aprende un selector nuevo; no continúa aquel checkpoint PPO.

## Qué se conserva y qué se aprende

El núcleo v2 confirma una trayectoria entrante con dos capturas diferentes y calcula el cruce `y = by + vy * ((-0.12 - bx) / vx)`. Clasifica como central el intervalo de ±0.30 m; activa el centro al confirmar y los saltos cuando faltan como máximo 1.8 s. La primera decisión se mantiene y el límite activo sigue siendo 28 s.

El selector recibe nueve valores en el primer compromiso: cruce y tiempo de llegada previstos; posición y velocidad del balón; edad de la medición; y varianzas de velocidad en ambos ejes. La normalización se ajusta exclusivamente con entrenamiento. Tres términos derivados del cruce y la llegada completan doce características. Seis cabezas logísticas estiman la posibilidad de atajada para cada defensa; la regularización y el margen de preferencia por la defensa original se seleccionan con validación.

El contexto no incluye el nombre del estilo, la semilla, el objetivo del tirador, la posición verdadera del balón ni la orientación del rival. Estos datos pueden aparecer en los registros de evaluación, separados de los valores que recibe el modelo. La orientación del rival se sigue midiendo en el pipeline, pero no se utiliza para seleccionar la defensa.

El instante de compromiso permanece congelado. Por tanto, elegir un salto donde v2 había decidido una guardia central puede ejecutarlo antes del instante habitual de ese salto. La prueba compara precisamente esta variante; las seis defensas no se presuponen intercambiables en alcance o duración.

## Recopilación y particiones

Cada contexto de entrenamiento o validación repite el penal completo seis veces, con la misma semilla física y una defensa distinta. La cabeza, la percepción y la predicción son iguales hasta el compromiso. El orden de las defensas rota para distribuir los efectos de arranque.

Un grupo sólo sirve para ajustar si las seis repeticiones tienen contexto y compromiso equivalentes, dentro de una tolerancia de `1e-9`, y ninguna termina en `sim_error`. Se comprueban el lado del núcleo, los tiempos, las nueve mediciones finitas y el nombre de la defensa realmente ejecutada. Los grupos sin compromiso, discordantes o fallidos conservan sus registros y no aportan etiquetas al ajuste.

El objetivo versionado es `keeper_defense_outcome_v1`: atajada = 1, cualquier otro resultado físico = 0. Un fallo del simulador nunca se convierte en una etiqueta negativa. En la evaluación final, fallos, goles y casos sin compromiso permanecen en el denominador.

| Partición | Base de semillas | Casos por estilo | Casos de los ocho estilos | Defensas/políticas por caso | Duelos usados |
|---|---:|---:|---:|---:|---:|
| Entrenamiento | 74001 | 12 | 96 | 6 defensas | 576 |
| Validación | 75001 | 6 | 48 | 6 defensas | 288 |
| Prueba ciega | 76001 | 20 | 160 | 3 políticas | 480 |

La semilla física es `base * 1_000_000 + índice_del_estilo * 10_000 + índice_del_caso`. Las particiones y los estilos tienen semillas independientes. Dentro de cada caso, las defensas o políticas comparten exactamente la misma semilla y condiciones de reinicio.

Los ocho estilos son `center`, `mixed`, `corners`, `soft`, `late`, `feint`, `v1` y `v2`. **Los tiradores `v1` y `v2` son nuestros modelos entrenados**, ejecutados por el controlador de movimiento B-Human. No son un equipo rival B-Human autónomo.

### Entrenamiento recopilado

| Comprobación | Resultado |
|---|---:|
| Contextos totales | 96 |
| Contextos aptos para ajustar | 91 |
| Contextos sin compromiso | 5 |
| Duelos físicos | 576 |
| Duelos sin compromiso | 30 |
| Errores del simulador | 0 |
| Grupos con contexto/compromiso discordante | 0 |
| Etiquetas disponibles por defensa | 91 |

Estos recuentos describen la recopilación. La frecuencia de atajadas en entrenamiento no demuestra una mejora del modelo final. Los resultados reservados se presentan aparte.

Validación: 48 casos y 288 duelos; 43 contextos elegibles, cinco sin compromiso, cero errores y cero discordancias. Cada defensa recibió 91 etiquetas de entrenamiento y 43 de validación. Se comprobaron las 804 etiquetas elegibles contra la defensa realmente ejecutada y su resultado. La media y escala coinciden exactamente con los 91 contextos de entrenamiento.

Se ajustaron tres regularizaciones durante 1500 épocas y se compararon tres márgenes por cada una. Validación eligió **L2 = 0.1 y margen = 0.03**, con puntuación contrafactual de 39/48 (81.25 %). No es la prueba final física: los casos sin contexto reciben cero en ese cálculo. Sus recuentos de defensa también incluyen una asignación de referencia para casos sin compromiso; los recuentos finales de abajo corresponden a movimientos efectivamente solicitados.

## Protocolo reservado antes del ajuste

El ensayo corto de integración había ejecutado el índice 0 de la base de prueba para `center` y `mixed`. El [protocolo ciego](heldout_protocol.json), fijado el **2026-10-11 a las 04:12:48 UTC**, excluye el índice 0 en los ocho estilos y usa exactamente los índices **1–20**. La selección quedó fijada antes del ajuste y de los resultados de producción; no se permite cambiar pesos usando esta prueba.

La evaluación inicial conserva su reporte de índices 0–19. El reporte ciego reutiliza 152 casos ya ejecutados de índices 1–19 y añade ocho casos del índice 20, sin alterar pesos ni el reporte anterior. Así conserva veinte casos por estilo y compara las mismas 160 situaciones para las tres políticas.

La corrida completa requiere **1368 duelos físicos de producción**: 864 para entrenamiento/validación, 480 en la evaluación inicial y 24 para los ocho casos adicionales. De ellos, 24 duelos del índice 0 quedan excluidos de la prueba ciega. Los **30 duelos del ensayo corto** se registran aparte y no se suman a las métricas finales.

La comparación final utiliza `original` —v2 con sus tres defensas—, `balanced` —la variante de cuatro defensas con centro espejado— y el selector aprendido de seis opciones. Se calcula la tasa macro dando el mismo peso a cada estilo. Las diferencias de atajadas son pareadas por caso; los intervalos de confianza se obtienen mediante bootstrap pareado independiente dentro de cada estilo. También se informa el estilo con peor tasa, las defensas seleccionadas, los errores y los casos sin compromiso.

## Resultados de la prueba ciega

Fuente: [heldout_report.json](heldout_report.json), con referencia SHA verificada a [selected_case_rows.json](selected_case_rows.json). El reporte inicial [test_report.json](test_report.json) se conserva intacto; sus índices 0–19 no sustituyen esta comparación 1–20. Se reutilizaron exactamente 152 casos y se ejecutaron ocho adicionales. En los 160 casos, las tres políticas compartieron contexto, primer compromiso e instante de activación exactamente iguales.

| Política | Atajadas / 160 | Tasa macro | Peor estilo y tasa | Errores | Sin compromiso |
|---|---:|---:|---|---:|---:|
| V2 original | 128 | 80.000 % | Esquinas, 65 % | 0 | 12 |
| Balanced | 134 | 83.750 % | Esquinas/amagos, 75 % | 0 | 12 |
| Selector aprendido | 135 | 84.375 % | Amagos, 75 % | 0 | 12 |

| Comparación pareada | Diferencia de tasa macro | Intervalo del 95 % | Interpretación |
|---|---:|---|---|
| Selector frente a v2 original | +4.375 pp | 0.000 a +9.375 pp | Señal favorable; el intervalo toca empate |
| Selector frente a balanced | +0.625 pp | −3.125 a +4.375 pp | No demuestra mejora adicional |
| Balanced frente a v2 original | +3.750 pp | +1.250 a +6.875 pp | Mejora en esta muestra: seis ganancias y cero pérdidas |

Los intervalos usan 10 000 remuestreos pareados, independientes dentro de cada estilo, con pesos iguales por estilo. Son intervalos bootstrap de esta muestra; veinte casos por estilo no garantizan el resultado contra otros tiros. No comparar directamente estos porcentajes con los 400 penales de versiones anteriores y otras semillas.

| Estilo, 20 casos | V2 original | Balanced | Selector |
|---|---:|---:|---:|
| Centrales | 17 | 17 | 16 |
| Mixtos | 17 | 18 | 18 |
| Esquinas | 13 | 15 | 16 |
| Suaves | 17 | 18 | 18 |
| Espera antes del tiro | 15 | 16 | 16 |
| Amagos | 14 | 15 | 15 |
| Nuestro tirador v1 | 19 | 19 | 20 |
| Nuestro tirador v2 | 16 | 16 | 16 |

| Defensa del selector | Casos |
|---|---:|
| `center` | 0 |
| `center_right` | 53 |
| `wide_left` | 1 |
| `wide_right` | 0 |
| `left` | 67 |
| `right` | 27 |
| Sin defensa comprometida | 12 |

Las seis opciones están habilitadas y recibieron etiquetas físicas; el modelo eligió cuatro en esta prueba. No se exige uso uniforme. Las guardias anchas no explican una mejora general: sólo se solicitó una ancha izquierda y ninguna derecha.

**Recomendación:** conservar v2 como núcleo y referencia. Para esta simulación, `balanced` es la alternativa sencilla que captura casi toda la ganancia y no perdió ningún caso atajado por v2. El selector entrenado queda experimental: frente a `balanced` mejora un caso de esquinas y uno del tirador v1, pero pierde uno neto central y empata en los demás estilos. Su peor tasa es 75 %, igual que `balanced`. Estos datos no justifican sustituir la regla simple por el modelo aprendido ni modificar sus pesos usando la prueba ciega.

## Procedencia y alcance

El [manifiesto de la corrida](run_manifest.json) identifica `keeper_selector_training_v1`, el contexto `keeper_defense_context_v1` y el modelo `keeper_defense_selector_model_v1`. Su fingerprint es `dc16dd0022ae2c10662fef1a1cdca1471f8a48ddfbb469824e4c823f85d2b9fa`. La captura parte del commit `5d38ba67c9c6aabddd26b0616aab5f321030528d`, con cambios locales declarados, y registra 103 hashes de fuentes, modelos y ejecutables.

Se verificaron los 559 archivos originales de `baseline/bhuman` contra su manifiesto, además del modelo RSI3 y los recursos físicos. El entorno registrado usa Python 3.11.9, NumPy 2.4.6, MuJoCo 3.3.7, Gymnasium 1.4.0 y Pillow 12.3.0. El selector se ajusta e infiere con NumPy; no necesita ejecutar PPO ni aprender movimientos articulares.

Los checkpoints por caso son atómicos y quedan ligados a la configuración, fuentes y semillas. La evaluación también vincula el SHA del modelo congelado. Cambiar el número de workers permite reanudar; cambiar fuentes, contratos, datos de partida o ajustes invalida el cache. Los dos módulos nuevos de prueba ciega añaden sus hashes a su reporte sin modificar el manifiesto ni los checkpoints originales.

Modelo: [keeper_selector_v1.json](../../../../models/vision/keeper_selector_v1.json), SHA-256 `1dfe42fabccd5ec775ef961d573b4370c5a935ef94d5433bdafd03f4bd356f7c`. Coincide antes y después de la prueba. Git conserva sus bytes exactos, igual que los datos y reportes, para preservar las referencias SHA. [artifacts.json](artifacts.json) registra las huellas de los archivos del paquete; [checks.json](checks.json) conserva la verificación de fuentes, etiquetas y trazas.

La percepción del portero utiliza RGB renderizado, detección B-Human del robot y sensores propios simulados de encoders, IMU y FSR. Sus lecturas son ideales y la odometría se estima mediante contactos. La física se pausa mientras procesa imágenes. El tirador utiliza observación privilegiada de la escena con `PerceptionNoise v1`. Todo el ensayo corresponde a MuJoCo; no valida cámaras, latencias ni movimientos en un NAO físico.

## Reproducción

Ejecutar desde la raíz del repositorio, con los runtimes originales ya compilados. Primero se recopilan datos; después se fija el protocolo ciego antes del ajuste:

```powershell
.venv/Scripts/python.exe tools/train_keeper_selector.py --stage collect --train-episodes 12 --validation-episodes 6 --test-episodes 20 --workers 6 --fit-epochs 1500 --output runs/keeper_selector/repro --model-output runs/keeper_selector/repro/model.json
.venv/Scripts/python.exe tools/evaluate_keeper_selector_heldout.py --output runs/keeper_selector/repro --init-protocol
.venv/Scripts/python.exe tools/train_keeper_selector.py --stage fit --resume --train-episodes 12 --validation-episodes 6 --test-episodes 20 --workers 6 --fit-epochs 1500 --output runs/keeper_selector/repro --model-output runs/keeper_selector/repro/model.json
.venv/Scripts/python.exe tools/train_keeper_selector.py --stage evaluate --resume --train-episodes 12 --validation-episodes 6 --test-episodes 20 --workers 6 --fit-epochs 1500 --output runs/keeper_selector/repro --model-output runs/keeper_selector/repro/model.json
.venv/Scripts/python.exe tools/evaluate_keeper_selector_heldout.py --output runs/keeper_selector/repro --model runs/keeper_selector/repro/model.json --workers 6
```

Una carpeta existente requiere `--resume` y los mismos ajustes. Para una nueva corrida independiente deben elegirse otra carpeta y otro archivo de modelo. `--init-protocol` no ejecuta motores y rechaza la inicialización posterior a `fit_report.json`.

## Ejecución, replay y comprobaciones

Se conservan [entrenamiento](train_dataset.json), [validación](validation_dataset.json), [ajuste congelado](fit_report.json), [log de entrenamiento/prueba](training.log), [log de la prueba ciega](heldout.log) y [log de pruebas unitarias](tests.log).

| Comprobación final | Estado / evidencia |
|---|---|
| Suite completa previa al módulo ciego | 252 pruebas, 102.039 s, OK; [tests.log](tests.log) |
| Pruebas adicionales del protocolo ciego | 14 pasan; comando y resultado en [checks.json](checks.json) |
| SHA del modelo antes/después de prueba ciega | Idéntico al ajuste congelado |
| Protocolo y reporte inicial preservados | SHA iguales; protocolo anterior al ajuste según declaración UTC y mtimes locales |
| 152 casos reutilizados y ocho adicionales | Exactos; 480 duelos seleccionados, cero errores |
| Ejecución headless y traza | Dos intentos `mixed`; el primero sin compromiso, el segundo ejecuta `center_right` y ataja |
| Replay del selector | 10 + 28 cuadros exactos; cero diferencias de acción o decisión |
| Predicción/activación frente a v2 | Decisiones completas y primer compromiso iguales; sólo cambia la defensa |

Los dos intentos siguen semillas consecutivas: [77001000000](watch/episode.json) terminó sin patada/compromiso; [77001000001](watch_77001000001/episode.json) ejecutó `center_right` a 2.700 s y terminó en atajada. Se paró en el primer compromiso, sin elegir por resultado. Se conservan ambos en [watch_attempts.json](watch_attempts.json); no intervienen en ajuste ni métricas ciegas. El segundo [replay](watch_77001000001/replay.json) confirma que v2 habría pedido `center` en el mismo cuadro, con cruce −0.000254 m y llegada prevista de 1.785875 s. Ese ejemplo no compara atajadas de dos políticas.

El helper archivado reproduce sin ejecutar física:

```powershell
.venv/Scripts/python.exe docs/validation/vision/keeper_selector_v1/replay_watch.py --trace docs/validation/vision/keeper_selector_v1/watch_77001000001/trace.jsonl --output runs/keeper_selector/replay.json
```

La GUI opcional `--live` no se probó manualmente en esta corrida. Las comprobaciones headless y de reproducción están separadas de las 160 situaciones que determinan el resultado reservado. Se añadieron dos episodios de runtime a los 1368 duelos de producción y 30 de integración. Los movimientos habilitados y el predictor base se describen en [DEFENSAS_PORTERO.md](../../../DEFENSAS_PORTERO.md).
