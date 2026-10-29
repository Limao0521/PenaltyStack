# Estado de la integración Copa con B-Human

> **Actualización:** el código B-Human del penal ya no vive en el fork sino en `bhuman/` de este repo (ver [bhuman/README.md](../bhuman/README.md) y la sección final). Las menciones al checkout del fork de abajo describen el estado anterior.

El checkout `~/workspace/semillero/SabanaHerons2026` contiene trabajo local sin commit para los escenarios `CopaPenaltyStriker` y `CopaPenaltyKeeper`, visibles en DeployDialog, y la ubicación `CopaPenalty`. La ubicación cambia el radio de la pelota a 32,5 mm; el resto de su campo todavía procede de HSL y no representa la cancha Copa. No declarar preparado para competición ni para movimiento físico hasta completar las verificaciones de abajo.

El comportamiento local propuesto inicia desarmado. Cabeza trasera rearma; un silbato nuevo o toque frontal inicia; pecho para y enclava; se rearma de nuevo con cabeza trasera. El tirador solicita una patada izquierda y se detiene al detectar su finalización. El portero solicita únicamente `Dive::prepare`, que se mapea a `sitDownKeeper`, y enclava la decisión al detectar movimiento entrante. El límite interno es 28 s desde el silbato detectado o el botón: si se usa el botón más tarde, **no garantiza** el límite oficial de 30 s desde el silbato. El comando de parada solicita postura de pie; no se ha comprobado cómo responde durante una patada o mientras está sentado.

El binario Nao compiló y las pruebas aisladas del estado y la decisión de sentado pasaron. Esto **no demuestra** que la configuración arranque en el robot, que el silbato llegue al comportamiento, que el botón detenga de inmediato una fase de movimiento, que la pelota se detecte en duela, ni que la postura de pie o sentado sea segura. El arranque de B-Human puede solicitar una postura incluso antes de activar el tiro: sostener el robot y comprobar primero. No se desplegó el binario.

En observación por Ethernet, las dos cámaras de ospinao respondieron en `192.168.49.7:7777` y `:7778`. Para verlas en ensayos por cable con el visor existente:

```bash
cd ~/workspace/semillero/BallDetectorTL
.venv/bin/python 15_watch_nao.py --ip 192.168.49.7 --camera dual --no-monitor
```

La versión actualizada de DeployDialog inicia ese visor para cada robot cuando se seleccionan tanto `Camera stream (upper + lower)` como `Restart bhuman`. Hay que cerrar una ventana antigua de DeployDialog y abrir la nueva para usar esta función. Si falla el arranque del visor, revisar `/tmp/nao-camera-view-<ip>.log`; el visor no es necesario para que el robot tome decisiones.

En ospinao el modelo configurado es `yolo_ball_320.onnx`, con clase `trionda` y SHA-256 `095cb322cbffb6ef8dff57a1f071dfa585b8525d9311180bb9b66335fec9fe0e`. Es diferente tanto del modelo 320 registrado en el último commit de este checkout (`a4ca5bae…`) como del artefacto anterior documentado como `Trionda Final Model` (`8fbe74e0…`). No se ha establecido con estos datos cuál fue el modelo efectivamente usado en RoboCup; el archivo local 320 está modificado sin commit. No sustituirlo ni afirmar equivalencia sin verificar el historial del torneo. La detección real requiere una prueba con pelota visible.

Pendiente: validar la carga real del escenario y la ubicación, la configuración de DeployDialog que pueda alterar el parámetro de salto, el arranque por silbato y botón con falsos positivos y señales viejas, el ABORT durante cada fase, visión y rango con la pelota real y duela, contacto único de patada, la solicitud y ejecución de `sitDownKeeper` sin salto, efectos de stream sobre percepción y control, y reinicio repetible. Ensayar progresivamente con el NAO sujeto y luego libre; respetar las instrucciones de seguridad del proyecto. El reglamento prohíbe redes inalámbricas durante la prueba oficial.

## Continuación: 27 de octubre

- Ospinao `192.168.49.7` se desplegó con `CopaPenaltyStriker`, ubicación `4v4_Full`; ambos puertos CAMF respondieron. Las últimas modificaciones locales no se han desplegado automáticamente.
- El modelo inferior documentado como Trionda Final (`8fbe74e0…`) se añadió como `trionda_copa_lower.onnx`, sin sustituir el modelo superior. En un frame inferior con balón completo produjo confianza aproximada 0,338 offline, pero el robot publicado informó `notSeen`: la detección real **sigue sin resolverse**. Se añadieron mensajes `[CopaBall]` para distinguir modelo cargado, tracking, antigüedad y proyección tras el próximo despliegue; verificar que los mensajes aparecen en `/home/nao/logs/bhumand_*.log` y comparar con la imagen. La comparación offline usa JPEG y no reproduce exactamente el procesamiento YUYV del robot.
- El portero o la rutina de tres sensores se sentaban y podían volver a ponerse de pie: se añadió un latch de apagado hasta que B-Human entre en unstiff y se reactive explícitamente, y se mantiene la guardia sentada del portero después de terminar el tiro hasta rearmar o apagar. Requiere prueba física incremental; mantener pulsados los tres sensores hasta entrar en unstiff para permitir el cierre normal del log.
- El visor `tools/copa_camera_viewer.py` muestra rectángulos y cuenta de **candidatos azules verticales**; no es una detección fiable de portería, no alimenta el comportamiento onboard y puede marcar objetos azules ajenos a la meta. Las pruebas son sintéticas; requiere imágenes de postes reales y evaluación en condiciones de campo. Los recuadros amarillos de BallSpots se quitaron y los marcadores de spots se apagan por defecto.
- El detector de robots existente tampoco se ha validado ni conectado a la toma de decisiones de Copa. El tirador sigue usando dirección de patada fija; no afirmar que apunta a postes o evita al portero basándose en visión.
- El visor necesita el entorno Python de `~/workspace/semillero/BallDetectorTL/.venv` (OpenCV y NumPy). Su copia versionada se basa en el visor local de ese proyecto, que no tiene remoto Git configurado. Para que DeployDialog lo abra, usar el script actualizado `Make/Common/openCameraViewer` del fork B-Human.

## Visión Copa y código en PenaltyStack (27 de octubre, segunda sesión)

**Organización.** El código B-Human del penal se trasladó a `bhuman/files/` (225 archivos sobre el commit base `0d1ea7ab` del fork). `tools/bhuman_overlay.py build` lo aplica a `~/workspace/semillero/SabanaHerons2026-penalty` y compila; el checkout principal del fork no se modifica. El commit que se había hecho en la rama `penaltystack-keeper-sit` se deshizo localmente (los archivos quedaron sin commit, como estaban); la rama remota hay que borrarla a mano.

**Balón.** Causa de "no detecta en la cámara inferior" (y del balón lejano perdido en la superior): los modelos dan confianza baja con la Trionda sobre duela y la luz del laboratorio (inferior 0,31; superior 0,29), por debajo de los umbrales (seguimiento 0,28 con dos frames; superior 0,35). El preprocesado del robot (YUYV→RGB) reproduce el resultado del PC (0,314 vs 0,338), así que no es un error de conversión. En frames sin balón ambos modelos dan ≤ 0,021. Cambios, sólo en escenarios Copa: umbrales `lowerConf 0,15`, `upperConf 0,20`, `kalmanInitConf 0,18`, `kalmanActiveConf 0,12`, `kalmanInstantInitConf 0,40`, se mantiene la confirmación de 2 frames; modelos fijados por cámara (`trionda_copa_upper.onnx` = el modelo 320 que estaba en ospinao, mejor que el del commit base: 0,289 vs 0,113; `trionda_copa_lower.onnx`); ubicación `CopaPenalty` (radio 32,5 mm) en `teams.cfg`. La muestra es pequeña (8 imágenes): validar falsos positivos en duela con gente y objetos alrededor. La solución de fondo es reentrenar con imágenes de la duela y la pelota naranja oficial (el visor guarda muestras con `t`/`n`).

**Postes azules y portería.** Módulo nuevo `CopaVisionPerceptor` en ambas cámaras: busca corridas verticales azules (YCbCr, Cb−128 ≥ 30 y Cb−Cr ≥ 45; poste del laboratorio Y=45 Cb=194 Cr=96), las agrupa en cajas altas y delgadas (≥ 18 % de la altura de imagen, alto ≥ 2× ancho) y descarta el travesaño y paredes porque siguen azules a los lados. Proyecta la base de cada poste al campo; si dos postes están a 1,5 m ± 25 % marca la portería (poste izquierdo/derecho, ancho, centro). En las 8 capturas reales encontró el poste en las 4 de la cámara superior y nada en las inferiores, incluida la del balón (que tiene parches azules). No hay captura con la portería completa: falta validar el par de postes y la distancia proyectada.

**Otros NAO.** Se reutilizan los detectores de robots del fork (`RobotDetector` arriba, `PlayersDeeptectorFeatBOPLower` abajo); `CopaVisionPerceptor` proyecta su base al campo y marca si están caídos. No se ha probado con un NAO delante.

**Retroalimentación.** `CopaCameraStreamer` (sólo escenarios Copa, mismos puertos y `cameraStreamer.cfg`) añade tras cada frame una sección `COPA` con balón en campo, postes, portería y robots. El visor dibuja lo que detecta el robot: postes en azul (amarillo si forman portería, con línea entre bases), NAO en naranja con distancia, y tres líneas de estado `BALON`, `PORTERIA`, `NAO`. Con un binario sin Copa vuelve a los candidatos azules calculados en el PC (rotulados "CANDIDATO PC"). En el robot, cada 5 s el log `/home/nao/logs/bhumand_*.log` tiene `[CopaVision]` (postes, portería, robots por cámara) y `[CopaBall]` (modelo, confianza, geometría).

**Verificado en el PC:** compilación Nao y DeployDialog desde el worktree; pruebas `tests/test_bhuman_overlay.py` (claves de parámetros completas, proveedores en hilos Upper/Lower, modelos presentes, ubicación Copa, sin lanzamiento, herramienta de overlay), `tests/test_bhuman_host_cpp.py` (estado de botones/apagado, selección de modelo, detector de postes con imágenes sintéticas), `tests/test_copa_goal_view.py` (protocolo COPA y frames CAMF sin sección); suite completa de PenaltyStack. **No verificado en el robot:** nada de esta sección se ha desplegado aún; ospinao dejó de responder antes de probar el visor contra el stream real.

**Otras observaciones en ospinao:** la cámara superior se corta 1–2 s cada pocos minutos ("no image… Terminating" y se recupera); un log de B-Human en `/home/nao/logs` ocupa 2,3 GB.

### Lista de pruebas (en orden)

1. `tools/bhuman_overlay.py build`; abrir la ventana de deploy del worktree; desplegar `CopaPenaltyStriker`, ubicación `CopaPenalty`, stream activado. Sostener el robot al reiniciar.
2. Arranque: ojos sin rojo; en el log aparecen `[CopaBall] camera=upper model=…trionda_copa_upper.onnx loaded=1` y lo mismo para `lower`, y `[CopaVision]` cada 5 s. El visor abre ambas cámaras con "deteccion: robot".
3. Balón, cámara inferior: Trionda quieta a 15–40 cm frente a los pies → `BALON` en verde con distancia coherente; mover el balón y comprobar que lo sigue; retirarlo → `BALON no` en ≤ 1 s.
4. Balón, cámara superior: a 1, 2 y 3 m; anotar a qué distancia deja de verse.
5. Falsos positivos: sin balón, con personas, pies, cables y el poste en vista durante 1 min → ningún `BALON`.
6. Postes: robot en la marca penal (1,3 m de la línea de gol + 20 cm) mirando la portería → dos `POSTE`, `PORTERIA ok ancho ≈1.50m`; anotar ancho y distancia medidos frente a los reales. Probar girando la cabeza o el robot ±30°.
7. Postes, falsos positivos: camisetas azules, el travesaño solo, objetos azules del laboratorio → sin `POSTE`.
8. NAO delante: portero sentado y de pie en la línea de gol → `NAO 1` con distancia ≈ 1,5 m desde el tirador. Repetir en `CopaPenaltyKeeper` con el tirador delante (≈ 1,5 m).
9. Botones y voz (ya implementados, sin probar en robot): cabeza trasera "striker mode arm", silbato o cabeza delantera "start", pecho "stop", tres sensores de cabeza se sienta, se queda sentado y guarda el log.
10. Portero: `CopaPenaltyKeeper`, tiro al arco → se sienta, **no se lanza**, y permanece sentado.
11. Tiro completo tirador vs portero con el visor abierto, y luego sin PC ni stream (la prueba oficial es sin red).

Pendiente después de validar: usar postes y portero para elegir el lado del tiro (el tirador sigue con dirección fija) y cambiar a la pelota naranja oficial.

## Tirador: balón → mirar portería → patear (27 de octubre, tercera sesión)

**Hallazgos en el robot.** Con los umbrales Copa la cámara inferior ya sigue la Trionda a ~20 cm. Agachado y con la cabeza nivelada se detectaron los dos postes, pero la portería no se emparejaba: la proyección al suelo dio el poste izquierdo a ~3,4 m y descartó el derecho (> 4 m), cuando la portería estaba a ~1,5 m (los postes ocupaban ~560 px de ancho). La distancia proyectada no es fiable; el **ángulo** (bearing) de cada poste sí. El detector de robots inferior marcaba la Trionda como `NAO`.

**Cambios.**
- `CopaVisionPerceptor`: bearing (ángulo respecto al robot) para cada poste y robot; portería = postes extremos separados entre 25° y 95°, distancia estimada con el ancho de 150 cm; descarta cajas de "robot" alrededor del balón (≤ 2,5 diámetros). Publica los bearings de la cámara superior para el comportamiento (`CopaVisionShared`, sin tocar los hilos de otros escenarios).
- Tirador en `running` (`copaShot.cfg`):
  1. **Buscar balón** con la cabeza abajo; si está a más de 30 cm camina hacia él **sin girar** (mirando a la portería) hasta quedar 20 cm detrás.
  2. **Mirar portería**: se queda quieto, dice "looking at goal", cabeza a +25°, 0°, −25° (1,2 s cada una, inclinación 5°) y acumula bearings de postes y robots.
  3. **Decidir** (`CopaShotPlanner.h`, punto único a sustituir por RL): con dos postes, apunta 25 % del ancho hacia dentro del poste del lado con más espacio respecto al portero; sin portero, lado por defecto (`defaultSide = 1`, izquierda); con un poste, infiere el otro; sin postes, recto. Dirección limitada a ±20° (gira alrededor del balón, no camina de lado). Dice "aim left/right/center" o "no goal, straight" y escribe `[CopaShot]` en el log.
  4. **Patear** con `forwardFastLeftLong` hacia ese punto (guardado en odometría para que no cambie al acomodarse).
- Visor: protocolo COPA v2 con ángulos: `POSTE IZQ +27deg`, `PORTERIA ok 53deg ~1.50m`, `NAO +3deg`.

**Verificado en el PC:** compilación; pruebas del planificador (portero a la izquierda → derecha y viceversa, un poste, sin postes, robot girado 10°), claves exactas de `copaShot.cfg` y `copaVisionPerceptor.cfg`, protocolo v2. **Sin probar en el robot.**

### Pruebas del tirador (sin portero, luego con NAO de pie)

1. Desplegar `CopaPenaltyStriker`; balón en la marca, robot 20 cm detrás mirando la portería. Cabeza trasera → delantera.
2. Debe decir "looking at goal", mover la cabeza izquierda-centro-derecha y decir "aim left" (sin portero). En el log: `[CopaShot] ... goal=yes keeper=no side=1 direction=…`. Si dice "no goal, straight": revisar `[CopaVision] camera=upper` durante el barrido y ajustar `headTilt`/`sweepPan` en `copaShot.cfg`.
3. Patea hacia dentro del poste izquierdo sin desplazarse de lado.
4. Repetir con un NAO de pie en la línea de gol, desplazado ~20 cm a la izquierda → debe decir "aim right"; desplazado a la derecha → "aim left". En el visor, el NAO aparece como `NAO ±Ndeg` y el balón ya no como NAO.
5. Robot colocado girado ~10° respecto a la portería → la dirección compensa (dentro de ±20°).
6. Balón a 1 m: camina recto hasta quedar detrás del balón y luego hace el barrido.

## Portero: seguir, caminar al balón y genuflexión B-Human (28 de octubre)

**Hallazgo.** El portero usaba `Dive::prepare` → keyframe `sitDownKeeper` (sentado con brazos abiertos), y `Dive.cpp` convertía **cualquier** petición de dive en ese sentado. El portero de SabanaHerons en RoboCup (`HandleGoalkeeperCatchBall`) bloquea con `InterceptBall` usando `genuflectStandDefender` (`squatWideArmsBack`, sentadilla con brazos atrás que vuelve a pararse). En el log, el portero veía la Trionda arriba con confianza 0,43–0,56 a ~1,3 m.

**Cambios.**
- `Dive.cpp`: sin `keeperJumpingOn`, deja pasar el sentado y las genuflexiones; los saltos laterales se convierten en la genuflexión del mismo lado. Nunca se lanza.
- `CopaKeeperPlanner.h` + `copaKeeper.cfg` (sólo `CopaPenaltyKeeper`), con señales relativas al robot de B-Human (`intersectionPositionWithOwnYAxis`, `timeUntilIntersectsOwnYAxis`), sin localización:
  - sin amenaza: de pie, sigue el balón con la cabeza;
  - balón hacia el portero (≥ 150 mm/s, cruza en ≤ 4 s), confirmado 150 ms: **camina de lado** hasta el punto de cruce, dentro de ±70 cm del inicio (área chica);
  - cuando llega (≤ 0,8 s o ≤ 50 cm) y no viene al cuerpo (> 8 cm): termina el paso, dice "block" y hace **`genuflectStandDefender`** hacia ese lado, sostenida ≥ 2,5 s; luego se levanta (keyframe de B-Human);
  - balón al cuerpo: se queda de pie.
- `[CopaKeeper]` en el log en cada cambio de acción (posición/velocidad del balón, cruce, desplazamiento).
- `teams.cfg` queda como archivo gestionado por la ventana de deploy: el overlay ya no lo pisa ni marca conflicto.

**Verificado en el PC:** compilación; pruebas del planificador (balón quieto o ruido → nada; tiro lento → camina y luego genuflexión; tiro rápido → genuflexión inmediata; al cuerpo → de pie; límite lateral; balón alejándose → nada) y que la ruta del portero nunca pide salto. **Sin probar en el robot.**

### Pruebas del portero (no hace falta tirador)

1. Desplegar `CopaPenaltyKeeper`; portero en la línea de gol entre postes; balón en la marca (1,3 m). Cabeza trasera → delantera.
2. Balón quieto 10 s: sigue el balón con la cabeza, **no** se agacha.
3. Rodar el balón **despacio** 40 cm a un lado: camina de lado y, al llegar, "block" + genuflexión hacia ese lado; se levanta solo.
4. Tiro **rápido** a una esquina: genuflexión inmediata hacia esa esquina, sin caminar.
5. Tiro al cuerpo: se queda de pie.
6. Revisar `[CopaKeeper]` en `/home/nao/logs/bhumand_*.log` después de cada tiro.

### Portero: genuflexión en simulación y caminar hacia el balón (28 de octubre, segunda prueba)

**Prueba en robot:** el portero sólo seguía el balón con la cabeza. El log no tuvo ninguna línea `[CopaKeeper]`: la decisión nunca dejó "watch", es decir, nunca se detectó el balón viniendo hacia él. La velocidad de `BallModel` (con 3–4 detecciones YOLO por segundo) no bastó para el disparo.

**Genuflexión en simulación** (`tools/record_keeper_block.py`, motor B-Human original y el mismo `keyframeMotionEngine.cfg`): `genuflectStandDefender` **no es un lanzamiento**, pero el robot **baja al suelo en un split lateral** (torso a 0,14 m, inclinación máx. 28° de cabeceo y 25° de alabeo, sin caída) y se levanta solo con `recoverFromSumo` en ~3 s. Vídeo y resumen en `runs/keeper_block/` (no versionados). En el simulador: `--mode manual --action block --block-side defender_left|defender_right`.

**Cambios:**
- `blockMotion` en `copaKeeper.cfg`: `0` de pie (nunca baja), `1` sentarse (`sitDownKeeper`), `2` genuflexión (**activa desde el 28 de octubre**, aprobada tras verla en simulación). Con `0` sólo camina a la línea del balón y bloquea con el cuerpo.
- El portero estima la velocidad del balón con sus propios avistamientos de los últimos 600 ms (mínimos cuadrados, en coordenadas de odometría para no confundir sus pasos con movimiento del balón) y calcula dónde y cuándo cruza su eje; usa la estimación de B-Human sólo si no hay ajuste propio.
- `[CopaKeeperBall]` cada 500 ms con posición, velocidad de B-Human, velocidad propia, cruce y desplazamiento lateral.

**Probar:** rodar el balón despacio a ~30–40 cm a un lado del portero (al cuerpo se queda quieto por diseño). Debe caminar de lado y quedarse de pie. Enviar el log con `[CopaKeeper]` y `[CopaKeeperBall]`.

### Portero: "dobla las rodillas y vuelve a pararse" (28 de octubre)

El log de esa prueba se perdió (B-Human reescribe `bhumand_(1).log` al reiniciar). Causa por código: el portero esperaba en postura alta (rodillas rectas), así que cada orden de caminar empezaba bajando a la postura de marcha; y el disparo se perdía entre detecciones porque exigía balón visto en los últimos 300 ms, cuando YOLO da 3–4 detecciones por segundo (`stale ageMs` 555–618 en el log). Entre detecciones volvía a "watch" y pedía postura alta.

Cambios: postura lista (rodillas flexionadas, como el portero B-Human) mientras corre; balón "visto" 700 ms; el disparo sigue activo 400 ms tras su último avistamiento usando la predicción (tiempo restante y cruce corregido por los pasos del propio portero); histéresis para caminar (empieza > 12 cm, se detiene < 8 cm). Nuevas claves en `copaKeeper.cfg`: `seenTimeoutMs`, `releaseMs`, `walkStartMargin`.

**Después de cada prueba, no reiniciar ni redesplegar antes de revisar el log.**

## Tirador: no tocar el balón en la marca antes de patear (28 de octubre)

**Problema:** el tirador tocaba el balón antes de patear. Con el balón a ~20 cm, `WalkToBallAndKick` tenía que desplazarse de lado y girar (hasta 20°) pegado al balón para llegar a la pose de patada de B-Human (`ballOffset` (−190, −50) mm de `forwardFastLeftLong`).

**Cambio:** nueva fase **align** entre "mirar portería" y "patear":
1. La posición del balón en la marca se memoriza en coordenadas de odometría al detenerse frente a él (el balón no se mueve hasta la patada).
2. Si el robot está más cerca que la pose previa, retrocede en línea recta.
3. Camina a la **pose previa**: la misma pose de patada que calcula `WalkToBallAndKickEngine`, 10 cm más atrás sobre la línea de patada; ahí hace el desplazamiento lateral y el giro, lejos del balón.
4. Con error ≤ 3 cm y ≤ 5° (o 8 s de límite) pasa a `WalkToBallAndKick`, que sólo avanza recto y patea.
5. `[CopaShot] aligned=… pre=(x,y,ángulo) ballFromMark=…mm` en el log: si `ballFromMark` es grande, el balón se movió antes de patear.

`copaShot.cfg`: `preKickClearance`, `alignTolerance`, `alignAngleTolerance`, `alignTimeoutMs`; `approachDistance` 400, `standOffDistance` 300.

## Distancia balón–pie y gemelo digital (28 de octubre)

**Distancia balón–pie.** Se modelan las dos plantas con `RobotDimensions` (cadera ±50 mm; planta 110 mm adelante, 50 atrás, 38,8 adentro, 50 afuera) y se calcula el hueco entre la superficie del balón (radio 32,5 mm) y el borde de planta más cercano. Durante la alineación: si el hueco es < `minFootClearance` (40 mm) el robot sólo retrocede; si es < 80 mm puede girar o desplazarse de lado pero no avanzar. Sólo la patada (`WalkToBallAndKick`) avanza hacia el balón. `[CopaShot] aligned` registra el hueco actual y el mínimo de la corrida.

**Gemelo digital** (`tools/copa_twin.py`). En cada ciclo el comportamiento publica lo que cree y decide (`CopaTwinShared`): estado y fase, pose del robot, balón, posición memorizada en la marca, postes, portero, punto de tiro, pose previa, a dónde camina, hueco balón–pie actual y mínimo, y cuánto se movió el balón. Todo en el marco del robot al pulsar inicio. Viaja en el stream de la cámara superior (COPA v3) y el visor lo dibuja al lado de las cámaras cuando arranca el tirador o el portero; cada corrida se guarda en `runs/twin/twin_<fecha>_<rol>_run<N>.jsonl`.

- En vivo: se abre con el visor (ventana de deploy) o `copa_twin.py --ip 192.168.49.7`.
- Después: `copa_twin.py --replay runs/twin/<archivo>.jsonl` (ventana) o `--video salida.mp4`; imprime resumen (lado, portería/portero vistos, hueco mínimo, balón movido antes de patear).
- Mapa: cancha Copa alineada a los postes vistos, marca esperada a 130 cm, plantas del robot (rojo si el hueco < 40 mm), rastro, pose previa (cian), flecha amarilla a donde camina, línea de tiro roja, portero "K".

Requiere el visor actualizado (protocolo COPA v3).

## Posición inicial conocida y reinicio del gemelo en cada parada (28 de octubre)

**Problema:** después del primer tiro, en los siguientes intentos el robot perdía la orientación.

**Cambio:** el reglamento fija la colocación, así que se usa como verdad de partida en cada corrida.
- Al pulsar inicio (`striker mode start`) se reinicia el marco de la corrida y se asume: robot en el origen mirando a la portería, balón en la marca a 20 cm (`startBallDistance`), línea de gol 130 cm más allá (`penaltyDistance`), postes a ±75 cm.
- Al detenerse (`striker mode stop`, fin de la patada o del tiempo) se vuelve a ese estado: se borran marca, decisión, alineación y holguras; el gemelo muestra "listo en posición inicial" y el grabador cierra el archivo de la corrida.
- El balón visto sólo reemplaza a la marca conocida si está a ≤ 15 cm de ella; si no, dice "ball not on the mark" y usa la marca conocida.
- La visión ya no define la portería: los postes vistos durante el barrido que estén a ≤ 12° de los conocidos dan el error de orientación de la colocación (mediana, máximo ±15°), y la portería conocida se gira esa cantidad. El portero visto decide el lado. Siempre hay portería; ya no existe "no goal, straight".
- `[CopaShot] postsMatched=… headingCorrection=…` en el log; el gemelo (COPA v4) muestra cuántos postes confirmaron la portería y el giro corregido.

`copaShot.cfg`: `startBallDistance`, `penaltyDistance`, `markMatchDistance`, `postMatchTolerance`, `maxHeadingCorrection`.

## Corrección: el gemelo se construye al ARMAR, anclado en el balón visto (28 de octubre)

La versión anterior reiniciaba al iniciar con distancias fijas. Flujo correcto, ahora implementado:

- **Armado (cabeza trasera):** nueva corrida y nuevo gemelo. El robot cree que mira a la portería y que **el balón que ve delante de los pies (cámara inferior) es la marca penal**: ése es el punto de partida del gemelo. Escaneo rápido sin moverse: (1) mira al suelo y toma la marca del balón visto (si no lo ve en 2 s, la supone 20 cm delante de las puntas y dice "ball not seen"; si lo ve dice "ball on the mark"); (2) barrido de cabeza: la portería se construye 130 cm más allá de la marca y los postes azules vistos la confirman y corrigen la orientación ("goal confirmed" / "goal not seen"); (3) queda mirando el balón.
- **Modo tirador (inicio):** conserva el gemelo del armado; vuelve a tomar la marca del balón visto delante de los pies, repite el barrido para decidir el lado (postes y portero; si no ve postes usa la corrección del armado), se alinea sin tocar el balón y patea.
- **Parada:** borra todo hasta el siguiente armado.

`copaShot.cfg`: `startBallFromToes` (marca supuesta si no ve el balón), `penaltyDistance`, `maxMarkDistance`, `ballScanMs`, `ballScanTilt`, `postMatchTolerance`, `maxHeadingCorrection`. Gemelo COPA v5 (marca vista/supuesta y fases del armado).

## Tirador: no moverse antes del barrido, caja detrás del balón, calibración de rango y gemelo en vivo (28 de octubre)

**Log de la prueba:** al armar, el balón se midió a 537, 397 y 468 mm del centro del robot cuando estaba ~310 mm (20 cm delante de las puntas): la proyección al suelo lee 100–230 mm de más (mismo sesgo que en los postes). Con "balón a más de 400 mm → caminar", el tirador caminaba hacia adelante **antes** del barrido y se desfasaba. Varias corridas terminaron en `aligned=timeout` con el balón movido 91–111 mm, y al perder el balón en la patada volvía a "buscar balón" y caminaba de nuevo.

**Cambios:**
- **Calibración de rango (altura de cámara + píxeles):** el rango es la altura de la cámara inferior sobre la tangente del ángulo bajo el horizonte; el ángulo está desviado. Al armar, como el balón está 20 cm delante de las puntas, se calcula el desvío de cabeceo (acumulado, máx. ±8°) y `YoloBallDetector` lo aplica a todo balón de la cámara inferior en escenarios Copa: el modelo de balón, el motor de patada de B-Human, la holgura pie–balón y el gemelo usan la distancia corregida. `[CopaShot] range calibration: measured=… corrected=… pitchOffset=…`.
- **Sin caminar antes del barrido:** en modo tirador sólo mira el balón, retoma la marca, barre y decide.
- **Caja detrás del balón** (marco del balón: origen en la marca, x hacia la portería): todo objetivo de ajuste se recorta para que las plantas nunca pasen la línea perpendicular por el balón (4 cm antes de su superficie, considerando el giro), con el centro del robot a ≤ 45 cm detrás y ≤ 15 cm a los lados, y giro ≤ 20°. Sólo la patada entra al balón.
- **Gemelo en vivo:** durante la alineación y la patada cada nuevo avistamiento corrige el 30 % de la diferencia entre el balón visto y la marca prevista (descarta > 15 cm); el mapa mantiene fija la cancha y mueve el robot corregido; muestra la línea del balón (roja), la zona permitida (verde), la calibración y la corrección acumulada (COPA v6).
- **Pie como B-Human:** izquierda con el pie derecho (`forwardFastRightLong`), derecha con el izquierdo.
- **Sin reinicios:** si pierde el balón en la patada vuelve a alinear, nunca a buscar caminando.

`copaShot.cfg`: `maxPitchCorrection`, `boxMaxBack`, `boxMaxLateral`, `trackGain`, `trackGate`; se quitaron `approachDistance` y `standOffDistance`.

## Balón naranja oficial (29 de octubre)

**Reglamento** (§ 3.3.4): pelota de plástico **naranja**, 65 mm, 55 g, sobre duela; líneas negras, postes azules, red blanca. Los modelos YOLO se entrenaron con la Trionda y no conocen una pelota naranja lisa. La duela es un naranja desaturado, así que el color solo no basta.

**Detector** (`CopaOrangeBall.h`, sin tipos de B-Human, probado en el PC): cada frame, en el hilo de la cámara (30 Hz, antes 3–4 Hz con YOLO), muestrea la imagen cada 2 px:
1. **Color relativo a la duela:** una muestra es naranja si Cr − Cb ≥ max(`minCrMinusCb` 50, mediana de la imagen + `floorMargin` 35), Cr − 128 ≥ 18, Cb − 128 ≤ −4 y el tono (Cr − 128)/(128 − Cb) está entre 0,35 (amarillo) y 1,9 (rojo). Duela del laboratorio (ospinao, 27 oct.): Cr − Cb mediana 7–20, percentil 99,9 ≤ 39. La franja roja de la Trionda mide 2,19 → descartada.
2. **Manchas conexas** (8 vecinos): redondas (ancho/alto ≤ 1,8 si no las corta el borde), llenas (≥ 0,42 de su caja; un disco da 0,785) y menores que 0,45 de la altura de imagen. Un balón cortado por el borde (pegado a los pies) conserva su extensión mayor como diámetro.
3. **Tamaño esperado:** el radio medido debe estar entre ½ y 2× el de una pelota de 65 mm en ese punto de la imagen según la matriz de cámara (con la corrección de cabeceo `CopaRange` en la inferior). Manos, camisetas, cinta y franjas pequeñas fallan aquí.
4. **Seguimiento:** el mismo Kalman de imagen de YOLO (con compensación del movimiento de cámara), 2 frames de confirmación (o confianza ≥ 0,8), sólo publica frames con candidato real.

**Integración** (`YoloBallDetector`): en escenarios Copa lee `copaOrangeBall.cfg` (archivo propio, para no tocar los `yoloBallDetector.cfg` de los otros 25 escenarios del fork). `mode = 2` (por defecto): si hay balón naranja seguido se usa; si no, YOLO (Trionda). `mode = 1`: sólo naranja, YOLO no se carga (menos CPU). `mode = 0`: sólo YOLO, como antes. El `BallPercept` es el mismo, así que modelo de balón, tirador, portero, gemelo y visor no cambian. Log cada 5 s: `[CopaOrange] camera=… floorCrMinusCb=… threshold=… orangeSamples=… blobs=… candidates=… best=… sizeRatio=… tracked=…` y `[CopaBall] … source=orange|yolo`.

**Calibración en PC** (`tools/copa_orange_calibrate.py`, entorno de BallDetectorTL): aplica la misma regla de color a capturas del visor (`s`) y muestra la máscara (`--show`); con `--ball X,Y,R` imprime estadísticas del balón y de la duela y el umbral máximo que conserva el 90 % del balón. La prueba de manchas y tamaño necesita la matriz de cámara y sólo corre en el robot.

**Verificado en el PC:** compilación Nao + DeployDialog; `Tests/CopaOrangeBallTest.cpp` (duela del laboratorio y duela barnizada, balón sombreado con brillo, balón cortado por el borde, balón lejano a ~4 m en la superior, dos balones, rechazo de mano, rojo, cinta naranja, tamaño incorrecto y punto sobre el horizonte); claves de `copaOrangeBall.cfg` = estructura; suite completa. En la captura real del laboratorio sin balón naranja la regla de color no marca nada en la inferior; en la superior sólo franjas de la Trionda y zonas pequeñas que la prueba de tamaño descarta. **No hay aún ninguna imagen real de la pelota naranja oficial; nada está probado en el robot.**

### Pruebas del balón naranja (en orden)

1. Con el visor abierto, capturar (`s`) la pelota naranja en la duela a 20 cm, 1 m, 2 m y 3 m y sin pelota; correr `copa_orange_calibrate.py <capturas> --ball X,Y,R`. Si «ball detected px» < 80 %, bajar `minCrMinusCb`/`floorMargin` al valor sugerido (sin bajar del percentil 99,9 de la duela) o ajustar `maxHueRatio` con el tono medido. Repetir en la duela del evento (5 min de calibración, § 2.10).
2. Desplegar `CopaPenaltyStriker`, ubicación `CopaPenalty`. En el log: `[CopaBall] … orange=2` en ambas cámaras.
3. Pelota a 15–40 cm frente a los pies → `BALON` verde, `[CopaBall] source=orange`; moverla → la sigue sin saltos; retirarla → `BALON no` en ≤ 0,3 s.
4. Cámara superior a 1, 2, 3 y 4 m: anotar desde qué distancia deja de verse (`[CopaOrange] candidates=0` o `sizeRatio` fuera de 0,5–2).
5. Falsos positivos 1 min sin pelota: manos, piel, camisetas rojas/naranjas, Trionda, cables, duela con sol → ningún `source=orange`. Si aparece, anotar `floorCrMinusCb`/`threshold` y subir `floorMargin`.
6. Portero (`CopaPenaltyKeeper`): rodar la pelota naranja; `[CopaKeeperBall]` debería tener muchas más muestras por segundo que con YOLO.
7. Para ensayar con la Trionda, dejar `mode = 2` (cae a YOLO) o `mode = 0`.

## Silbato como disparador (30 de octubre)

**Flujo:** cabeza trasera arma → el **silbato** del juez o la **cabeza delantera** inician el tiro (lo primero que llegue). Un silbato anterior al armado, o mientras el robot está parado, se ignora; con el tiro en curso, los dos disparadores no hacen nada. El límite de 28 s cuenta desde el silbato detectado (o desde el toque).

**Por qué no funcionaba:** el silbato nunca llegaba al comportamiento. Sin GameController el robot nunca pasa por *set* ni *playing*, y B-Human sólo escucha en esos estados por dos lados: `AudioProvider` vacía las muestras (`onlySoundInSetAndPlaying = true` en `Robots/Default`) y `WhistleRecognizer` sólo detecta en *set*/*playing*. La máquina de estados (`CopaPenaltyState`) ya aceptaba el silbato.

**Cambios** (sólo escenarios Copa):
- `Config/Scenarios/CopaPenalty*/audioProvider.cfg`: copia de `Robots/Default` con `onlySoundInSetAndPlaying = false` (la carpeta del escenario tiene prioridad sobre `Robots/Default`).
- `WhistleRecognizer.cpp`: en `CopaPenaltyStriker`/`CopaPenaltyKeeper` detecta en cualquier estado de juego. Los demás escenarios no cambian.
- `[CopaStart] trigger=whistle|headFront role=… whistleConfidence=… delayMs=…` en el log al iniciar (`delayMs`: retraso entre el silbato y la reacción).

**Limitación conocida:** B-Human no escucha mientras el robot habla (`soundWasPlaying`), para no detectar su propia voz. Si el juez pita justo durante "striker mode armed" o los avisos del barrido al armar ("ball on the mark", "goal confirmed"), ese silbato se pierde: hay que usar el botón delantero o volver a pitar. Armar con tiempo antes del silbato.

**Verificado en el PC:** compilación; `CopaPenaltyStateTest` (silbato o botón tras armar, reloj de audio ligeramente adelantado, silbato antes de armar ignorado, toque durante el tiro ignorado); configuración de audio con las mismas claves que `Robots/Default`; suite completa. **Sin probar en el robot.**

### Pruebas del silbato

1. Desplegar `CopaPenaltyStriker`. En el log, sin GameController: líneas `Whistle monitor:` al silbar (antes no aparecían fuera de *set*/*playing*).
2. Silbar **sin armar** → no pasa nada.
3. Cabeza trasera ("striker mode armed"), esperar a que termine de hablar y del barrido, silbar → "striker mode start", `[CopaStart] trigger=whistle`, `delayMs` < 1000.
4. Pecho (stop), rearmar, cabeza delantera → `[CopaStart] trigger=headFront`.
5. Con ruido de público, voces y aplausos 1 min tras armar → no inicia solo. Si inicia, revisar `whistleConfidence` y endurecer el perfil (`whistleRecognizer.cfg`: `mouthWhistleEnabled = false` deja sólo silbatos de árbitro).
6. Repetir 2–4 con `CopaPenaltyKeeper`.
7. Silbato real del evento (si es posible, el mismo modelo que usarán los jueces) a la distancia real del juez.

### Silbato: falsos disparos y "start" seguido de "stop" (30 de octubre, prueba en ospinao)

**Log** (`bhumand_(1).log`, 29 oct. 20:11–20:35): 7 inicios por silbato y 1 por botón. Las dos primeras corridas duraron 26 s y 9 s; desde la patada de 20:32:22 todas dijeron "striker mode stop" en el mismo segundo que "start", **también con el botón**. El balón naranja se vio bien: confianza 0,92–0,97, `sizeRatio` 1,1–1,2, balón a 310 mm (la marca).

**"start" → "stop" inmediato (error del cambio anterior):** al añadir `[CopaStart]` se borró `copaLastKick = theMotionInfo.lastKickTimestamp;`. El tiro termina cuando cambia la marca de tiempo de la última patada; sin reiniciarla, después de la primera patada cada tiro nuevo se daba por terminado al instante. Restaurada, con prueba que lo comprueba.

**Cualquier sonido agudo inicia:** el detector se abre tras `onsetConsec = 3` ventanas de 5 ms (~15 ms de tono): clics, chirridos y pitidos bastaban. En Copa (`copaWhistle.cfg`) una detección sólo cuenta como silbato tras **`minToneMs = 250` ms de tono real** (ventanas que pasan todos los filtros; la cola de `offFrames` no cuenta); `minChannels = 1` (2 exige ambos micrófonos). Perfil `mouth_whistle` (el más permisivo, 1,7–4,2 kHz) apagado en Copa: el juez usa silbato. Log: `[CopaWhistle] accepted profile=… toneMs=… peakHz=…` y `[CopaWhistle] too short … toneMs=… (min 250)`.

**Probar:** (1) armar y hacer ruidos agudos (golpes, chirridos de silla, aplausos, voz aguda) → ningún inicio; en el log `too short` con `toneMs` bajos. (2) Silbato corto normal del juez → inicia; anotar `toneMs` de `accepted`. Si un silbato real queda en `too short`, bajar `minToneMs` (p. ej. 180); si aún entra ruido, subirlo o poner `minChannels = 2`. (3) Patear y, tras "stop", rearmar e iniciar de nuevo → el segundo tiro debe correr normalmente. (4) Para probar silbando con la boca, `mouthWhistleEnabled = true` en `whistleRecognizer.cfg` del escenario.

## Menos CPU para el torneo y silbato por duración (30 de octubre, segunda prueba)

**Log y `top` en ospinao (20:44–20:49):** `procCPU` 315–350 % de 400 %, carga 5 en 4 núcleos, ~5 % libre; el robot decía "CPU temperature" cada ~25 s, y cada aviso hablado deja al detector de silbato sordo mientras suena. Hilos: dos `Upper` y dos `Lower` de 58–79 % cada uno (cámara + inferencia YOLO en segundo plano). Silbatos reales: `[CopaWhistle] too short … toneMs=30/35`.

**Quitado en Copa (no lo usa el penal):**
- **YOLO** (`copaOrangeBall.cfg`, `mode = 1`): sólo balón naranja; los dos hilos de inferencia no arrancan. Para ensayar con la Trionda, `mode = 2`.
- **Logger binario** (`logger.cfg`, `enabled = false`): codificaba en JPEG cada frame de ambas cámaras y escribía ~1 GB cada 6 min. El log de texto `bhumand_*.log` sigue igual.
- **Percepción de cancha HSL** en ambas cámaras: líneas, círculo, intersecciones, marca penal, regiones de scan lines, ball spots de la Trionda.
- **Redes de la cámara inferior**: BOP (segmentación) y PlayersDeeptector (robots), y su límite de campo.
- Módulo `CopaPerceptionStub` entrega esas representaciones vacías ("nada visto"), así ningún módulo que aún las requiera (autolocalización, modelo de obstáculos, visor) queda sin proveedor (eso detiene B-Human). Comprobado con una verificación estática del grafo de módulos: los mismos resultados que la configuración anterior, que sí arranca.

**Se mantiene:** cámaras y matrices, contorno del cuerpo, balón naranja, postes/portería (`CopaVisionPerceptor`), detector de robots de la cámara **superior** con su límite de campo y clasificador de camisetas (el tirador elige el lado según el portero), visor (sólo codifica si hay un cliente conectado).

**Silbato:** el tono se mide como tramo desde el inicio hasta el último frame que pasa todos los filtros (un silbato de árbitro tiene trino y algunos frames fallan dentro); `minToneMs = 200`. Un clic sigue midiendo ~15 ms.

**Probar:**
1. Desplegar; ojos sin rojo (si hay rojo, enviar el log: sería un proveedor faltante).
2. `[ResourceMonitor] procCPU` en reposo y armado, comparado con 315–350 %; sin "CPU temperature".
3. Balón naranja abajo y arriba como antes (`source=orange`); `[CopaVision] camera=upper … robots=` con el portero delante.
4. Silbato: `accepted … toneMs=` ≥ 200 inicia; clics y chirridos `too short`. Ajustar `minToneMs` con los valores medidos.
5. Tiro completo: alinear, patear, "stop"; rearmar y repetir.

### Silbato: duración medida 6× corta (30 de octubre, tercera prueba)

**Log (20:59–21:02):** CPU 105–122 % (antes 315–350 %), sin "CPU temperature". El monitor vio silbatos fuertes que pasan todos los filtros (`pMax` ~2000 frente a 41, `snr` 11–21, `fail=[] passCount=9`) pero ninguno inició.

**Causa:** el reconocedor analiza **una vez por bloque de audio** (`frames = 512` a 16 kHz ≈ 32 ms), pero su FSM avanza el contador de muestras sólo un salto de 5 ms (80 muestras) por análisis. La duración del tono se calculaba con ese contador: cada silbato medía ~6,4× menos (los 30–35 ms del log anterior eran ~200 ms reales), y con `minToneMs = 200` hacía falta ~1,3 s de silbato continuo. Además, las líneas `too short` compartían el límite de 1 s de las líneas del monitor y no aparecían.

**Cambio:** la duración se mide con el reloj del hilo de audio (`FrameInfo`), del primer al último análisis que pasa todos los filtros; `minToneMs = 150` (clics ~60–100 ms, silbato de árbitro ≥ 250 ms); `too short` con límite propio de 300 ms.

**Probar:** armar → silbato corto normal → `[CopaWhistle] accepted … toneMs=` y "striker mode start". Golpes/chirridos → `too short toneMs=` < 150. Enviar los `toneMs` de ambos para ajustar el umbral.
