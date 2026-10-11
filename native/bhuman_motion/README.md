# Runtime motriz original B-Human

Fuentes originales: `baseline/bhuman`, revisión SabanaHerons2026
`2360ab1100ad28f70e190c893f6b095651ff0bbc`. Los hashes de bytes originales
se conservan en `baseline/motion_manifest.json`. El checkout de origen no
interviene en la compilación ni ejecución posterior.

Desde la raíz del proyecto, con Visual Studio 2022 o 2026 Build Tools y CMake:

```powershell
.\.venv\Scripts\python.exe tools/build_motion_runtime.py
```

El ejecutable es `native/bhuman_motion/build/Release/bhuman_motion.exe`.
El script obtiene Eigen 3.4.0, nlohmann JSON 3.11.3 y únicamente el header
de declaraciones del protocolo HSL v20, verificando SHA256 contra
`dependencies.lock.json`. No compila ni inicia un GameController.

## Qué se ejecuta

Cada proceso posee el Blackboard original. Se instancian las clases originales
de módulos y se invocan los handlers de `getModuleInfo`, que llaman a sus
`update` originales. MotionEngine gestiona las fases originales de
WalkingEngine, KickEngine, KeyframeMotionEngine y FallEngine. También se ejecutan
WalkAtSpeedEngine, WalkToPoseEngine, WalkToBallEngine, WalkToBallAndKickEngine,
WalkKickEngine, WalkLearnerProvider, DribbleEngine, ReplayWalkRequestProvider,
HeadMotionEngine, ArmKeyFrameEngine, PointAtEngine y EnergySavingProvider.

Los proveedores originales de FSR, apoyo, contacto con suelo, estado del gyro,
modelo cinemático, matriz de torso, estado de caída, holgura y estabilidad
reciben sensores actuales. No hay reproducción de trazas ni marcha generada
en el adaptador.

Las entradas de orientación de InertialData provienen de la orientación real
del torso en MuJoCo. `orientation2D` y `angle` siguen exactamente el contrato
de InertialDataProvider: `removeZRotation` y vector de rotación angle-axis
horizontal con z=0. Usar Euler impedía recuperar grandes inclinaciones.
JointAnglePred recibe los ángulos medidos, sin modelo neuronal
de predicción del desgaste mecánico. Corriente cero, temperatura 35 C,
GyroOffset calibrado y variancias cero son un perfil ideal explícito de
simulación. Las representaciones de percepción ajenas al ensayo (obstáculos,
calibraciones de cámara, botones y estado de partido) no ejecutan percepción.

## Adaptaciones de plataforma

Se preservan los originales. El script genera copias en `build/generated`:

- `MotionEngine.cpp` y `WalkToPoseEngine.cpp`: elimina únicamente el include y
  bloque de exportación de diagnósticos de RLSharedState POSIX. El cálculo
  del movimiento original se conserva; los diagnósticos salen por JSONL.
- `File.cpp`: `getBHDir()` retorna la raíz del baseline fijada en la compilación.
  Es necesario porque el ejecutable aislado no está dentro del árbol original
  que el mecanismo de búsqueda espera.

`host.cpp` implementa únicamente SystemCall de simulación, diagnóstico de
audio a stderr, el constructor explícito de Settings y la búsqueda de Config.
El host `ThreadFrame` sólo establece los punteros de Global mediante la
amistad que el framework original declara. No se enlazan threads de robot,
SimRobot, LoLA, percepción, comunicación de equipo ni servidor de debug.

### Calibración simulada `mujoco-truth-v2`

- `BallSpecification.radius` recibe el radio real del geom MuJoCo: **32.5 mm**,
  diámetro 65 mm. El Config original de Sabana tiene radio 65 mm y permanece
  intacto. B-Human de referencia usa [radio 50 mm](https://raw.githubusercontent.com/bhuman/BHumanCodeRelease/master/Config/Locations/Default/ballSpecification.cfg).
- El golpe derecho de `kickForwardFast` conserva todos los endpoints originales,
  pero su punto dinámico de fase 3 lleva `duration=200` ms, frente a los 100 ms
  del KMC. Un wrapper de `KickGenerator.createPhase` aplica ese parámetro; el
  KickEngine original calcula interpolación, equilibrio, IK y finalización.
  No se desplazan endpoints ni se modifican sensores para simular aciertos.
  La izquierda conserva sus 100 ms. La prueba previa midió un fallo de unos
  8 mm y retraso de rodilla derecha de unos 0.2 rad; el golpe más lento logra
  contacto sin caída. Es una calibración física de este modelo, no aprendizaje.
- `HELLO.provenance` registra perfil, calibración y SHA256 de las fuentes del
  host. METADATA registra además READY, radio y hashes de adaptadores Python.
  Para comparar RL, usar el mismo mundo y perfil; las fuentes originales y
  el informe anterior se conservan como referencia. No confundir este perfil
  calibrado con la configuración KMC sin adaptaciones.

## Protocolo JSONL 1

HELLO se emite al iniciar. INIT fija `trial`, `role`, `seed`, `protocol=1` y
`ball_radius_mm` (por defecto 32.5, rango >0 y ≤200), y responde READY con el
radio confirmado. STEP requiere `trial`, `sequence` empezando en 0 y
`time_ms` con incrementos de 12 ms. El primer tiempo puede ser cero o un
offset de simulación. Internamente se añade 100000 ms, como hace el reloj
original para no confundir cero con un evento inexistente.

Sensores: `angles[26]`, `velocity[26]`, `gyro[3]` en rad/s, `acc[3]` en m/s²
de fuerza específica (+g vertical en reposo), `orientation[4]` wxyz unitario
con yaw de mundo incluido, `fsr[8]` en kg equivalente. FSR se ordena izquierda
fl/fr/bl/br y derecha fl/fr/bl/br. Articulaciones siguen `HELLO.joint_names`
y el orden original Joints, incluyendo ambas manos y HipYawPitch derecho.

`request.motion` admite stand/walk/kick/block/getup. Marcha usa forward/lateral
en mm/s y turn en rad/s. Patada admite forwardFastLeft/forwardFastRight,
`kick_length_mm` entre 1500 y 3500 (por defecto 2500), y ball[2] en milímetros
del plano robot. El alcance solicitado alimenta KickLengthConverter original;
no garantiza esa distancia recorrida en MuJoCo. Block admite seis defensas:
center/center_right (genuflectStand y espejo), wide_left/wide_right
(genuflectStandDefender y espejo), left/right (keeperJumpLeft y espejo).
prepare solicita sitDownKeeper como preparación separada. HELLO anuncia
keeper_defenses_v2; el protocolo 1 y las tres solicitudes antiguas se conservan.
Detalle y validación en [DEFENSAS_PORTERO.md](../../docs/DEFENSAS_PORTERO.md).
Cabeza usa head_yaw/head_pitch
en radianes. Getup solicita stand: el motor original decide la recuperación
según FallDownState, sin forzar estados inventados. Después de que el salto
lateral entra en `keeperJumpLeft`, la sesión solicita getup/stand una vez.
`KeyframePhase::isDone` mantiene el salto hasta su finalización o salida
original de fallo y sólo entonces permite la recuperación. La guardia central
se conserva mientras permanece solicitado el bloqueo.

OUTPUT devuelve ángulos, rigidez resuelta 0..100, odometría en mm/radianes,
nombre de fase original, petición, busy y pulso kick_completed. Off/ignore
se resuelven con la medida actual antes de enviar al servo. Nunca se
aplican valores no finitos. ERROR termina el proceso; reiniciar exige
crear uno nuevo. SHUTDOWN termina ordenadamente.
