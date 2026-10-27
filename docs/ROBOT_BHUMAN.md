# Prueba en robot real bajo B-Human (SabanaHerons2026)

## Por qué esta ruta existe

Los NAO v6 del laboratorio (ronaoldinho, ospinao, …) corren una imagen sin
NAOqi: no hay `python2` ni módulo `qi`, y el proceso `bhuman` (servicio de
usuario `bhuman.service`) es dueño exclusivo del hardware (`/dev/video0`,
`/dev/video2`, `/dev/snd/*` y las articulaciones). Nada puede ejecutarse "al
lado": el adaptador onboard `penalty/nao.py` de este repo sólo aplica a robots
con imagen NAOqi 2.8 y `qi` disponible. En los robots del laboratorio la prueba
de penal **vive bajo bhuman**: se implementa dentro del fork del equipo
`SabanaHerons2026` (B-Human 2023 + HSL), que es el código que ya se despliega
en ellos con `Make/Common/deploy`.

## Dónde vive el código

El código B-Human del penal está en `bhuman/files/` de este repo, fijado al
commit `0d1ea7ab` del fork (ver [bhuman/README.md](../bhuman/README.md)). No
se hacen commits ni ramas en `SabanaHerons2026`: `tools/bhuman_overlay.py`
copia esos archivos a un worktree limpio
(`~/workspace/semillero/SabanaHerons2026-penalty`) y compila ahí. Las rutas de
archivos citadas abajo son relativas al fork.

El contrato de PenaltyStack —visión del balón Trionda en ambos roles, patada
del tirador y portero que **únicamente se sienta, nunca se lanza**— se cubre
con el comportamiento de penal que el fork ya tiene, más un parámetro nuevo:

| Contrato PenaltyStack | Ruta bajo bhuman |
|---|---|
| Visión del balón (striker y keeper) | Detector Trionda del equipo (`yoloBallDetector`, YOLO + onnxruntime) → `BallModel`/`FieldBall` |
| Patada del striker | `HandlePenaltyShootout` → opción `PenaltyTaker` → `GoToBallAndKick` con `forwardFastLeft/RightLong` |
| Portero sólo se sienta | `PenaltyKeeper` → `InterceptBall` → skill `Dive` → keyframe original `sitDownKeeper` |

Cambios (mínimos y conmutables, el baseline del equipo no cambia por defecto):

1. `Src/Representations/Configuration/BehaviorParameters.h` — parámetro nuevo
   `(bool)(false) keeperSitWhenNoDive` (con default, por lo que los `.cfg`
   existentes que no lo mencionan siguen cargando igual).
2. `Skills/Output/MotionRequest/Dive.cpp` — con `keeperJumpingOn == false` y
   `keeperSitWhenNoDive == true`, toda petición de dive se convierte en
   `Dive::prepare`, que el `KeyframeMotionEngine` ya mapea al keyframe
   `sitDownKeeper` (definido en `Config/Robots/Default/keyframeMotionEngine.cfg`).
   Mientras el comportamiento siga pidiendo `Dive::prepare`, el keyframe se
   considera compatible y el portero **permanece sentado**; con los parámetros
   del equipo (`keeperSitWhenNoDive = false`) el comportamiento previo
   (quedarse de pie) no se toca.
3. `Options/PenaltyShootout/PenaltyKeeper.h` — `allowDive` ahora también es
   cierto con `keeperSitWhenNoDive`, para que la intercepción entre por la ruta
   `keeperSitJump` → `Dive` (y de ahí al sentado) en vez del fallback de pie
   `audioJump`. Durante penal (`allowGetUp = false` en shootout) no hay
   transición de salida: la guardia sentada se mantiene, igual que el latch de
   la simulación de PenaltyStack.
4. `Config/Scenarios/PenaltyStack/` — copia del escenario desplegado
   `4v4_Full` con `behaviorParameters.cfg`: `keeperJumpingOn = false`,
   `keeperSitWhenNoDive = true`, y `strategyBehaviorControl.cfg`:
   `enableEmbeddedPPO = false` (baseline puro B-Human, sin políticas RL ni
   modelos onnx de RL cargados).

El penal durante juego normal (`HandlePenaltyKick`, estado PENALTYKICK) usa la
misma opción `PenaltyKeeper`, así que el sentado aplica también ahí.

## Compilar y desplegar

```bash
cd ~/workspace/PenaltyStack
.venv/bin/python tools/bhuman_overlay.py worktree   # sólo la primera vez
.venv/bin/python tools/bhuman_overlay.py build      # aplica bhuman/ y compila Nao + DeployDialog
~/workspace/semillero/SabanaHerons2026-penalty/Make/Common/deployDialog Develop
```

En la ventana de deploy: escenario `CopaPenaltyStriker` o `CopaPenaltyKeeper`,
ubicación `CopaPenalty`, marcar `Camera stream (upper + lower)` y
`Restart bhuman`; el visor `tools/copa_camera_viewer.py` se abre solo.
Equivalente por consola:
`cd ~/workspace/semillero/SabanaHerons2026-penalty/Make/Common && ./deploy Develop <ip> -s CopaPenaltyStriker -l CopaPenalty -nc -b --camera-stream on --open-camera-viewer`.

`deploy` detiene `bhuman.service`, hace rsync del binario + `libonnxruntime` +
todo `Config/` a `/home/nao/Config`, fija escenario y ubicación en
`settings.cfg` y rearranca el servicio con `-b`. Para volver al código del
equipo, desplegar desde el checkout normal del fork.

## Procedimiento de prueba en campo

1. **GameController** (HSL): en el PC del semillero,
   `~/workspace/semillero/run_hsl_gamecontroller_sabana.sh`
   (competition `small_foundation`, equipos 49 y 5, interfaz `eno1` por
   defecto). Los `teamNumber` de `settings.cfg` de cada robot deben coincidir
   con home/away; se pueden fijar con `./deploy … -t <num> -p <jugador>`.
2. Colocar portero en su área y tirador detrás de la marca penal con el balón
   Trionda en la marca.
3. Llevar el juego a penal: **penalty shootout** (estados
   INITIAL→SET→PLAYING; en PLAYING el equipo atacante ejecuta `PenaltyTaker` y
   el portero rival `PenaltyKeeper`) o un **penalty kick** durante juego
   normal.
4. Observables esperados:
   - Tirador: busca el balón con la visión Trionda, camina a él
     (`penaltyStrikerWalkSpeed = 0.5`) y patea `forwardFast*Long` hacia una
     esquina (usa obstáculos si `penaltyStrikerUseObstacles = true`).
   - Portero: de pie en guardia; cuando el balón se mueve hacia el arco
     (`penaltyKeeperShouldCatchBall`) ejecuta el keyframe `sitDownKeeper` y
     permanece sentado — **no hay salto ni lance** (`keeperJumpingOn = false`).
5. Verificación en el robot: `ssh nao@<ip>` y revisar
   `/home/nao/Config/settings.cfg`, `journalctl --user -u bhuman.service` o
   los logs en `/home/nao/logs`. La detección del balón puede validarse con el
   streaming web del fork (cameraStreamer) o grabando un log y viéndolo en
   SimRobot.

## Relación con este repo

PenaltyStack sigue siendo el baseline de simulación/verificación (MuJoCo +
motor B-Human original): el contrato "portero sólo se sienta" está probado ahí
contra el keyframe `sitDownKeeper` original (`tests/test_sim_acceptance.py`),
y la visión Trionda contra verdad de cámara (`tools/vision_check.py`). En los
robots del laboratorio, la ejecución real la hace bhuman con los cambios de
esta página; el paquete NAOqi (`deploy/package.py`, `deploy/deploy.py`,
`penalty/nao.py`) queda reservado a unidades con imagen NAOqi 2.8.
