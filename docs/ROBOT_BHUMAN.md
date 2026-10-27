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

## Qué se implementó en el fork (rama `penaltystack-keeper-sit`)

Al 27 de octubre de 2026, estos cambios están sin commit en el checkout local
del fork; la rama local por sí sola no permite reproducirlos desde remoto.
La compilación y el despliegue siguientes requieren ese checkout con sus
cambios y el escenario `PenaltyStack` presentes.

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

## Compilar y desplegar (desde el checkout del fork)

```bash
cd ~/workspace/semillero/SabanaHerons2026

# Compilar el binario del robot (toolchain V6 ya configurada en Build/Linux/CMake)
( cd Make/Linux && cmake --build ../../Build/Linux/CMake/Develop --target Nao --config Develop )

# Desplegar a ronaoldinho (Ethernet 192.168.49.2) y reiniciar bhuman
cd Make/Common
./deploy Develop 192.168.49.2 -s PenaltyStack -nc -b
```

`deploy` detiene `bhuman.service`, hace rsync del binario + `libonnxruntime` +
todo `Config/` a `/home/nao/Config` (por eso el binario vive ahí), fija
`scenario = PenaltyStack;` en `settings.cfg` y rearranca el servicio con `-b`.
Usa la llave del equipo `Install/Keys/id_rsa_nao` (vía `Make/Common/setupSSH`).
Para volver al baseline del equipo: `./deploy Develop <ip> -s 4v4_Full -nc -b`
desde la rama que corresponda.

Despliegue realizado el 27 de octubre de 2026 en ronaoldinho: binario Develop
`30cc8cc8…` (prefijo SHA-256), escenario `PenaltyStack`, servicio
activo.

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
