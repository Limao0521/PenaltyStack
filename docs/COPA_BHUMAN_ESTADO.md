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
