# Estado de la integración Copa con B-Human

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
