# Runtime motriz original B-Human

Fuentes originales: `baseline/bhuman`, revisión SabanaHerons2026
`2360ab1100ad28f70e190c893f6b095651ff0bbc`. Los hashes de bytes originales
se conservan en `baseline/motion_manifest.json`. El checkout de origen no
interviene en la compilación ni ejecución posterior.

Desde la raíz del proyecto, con Visual Studio 2022 Build Tools y CMake:

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
del torso en MuJoCo. JointAnglePred recibe los ángulos medidos, sin modelo neuronal
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

## Protocolo JSONL 1

HELLO se emite al iniciar. INIT fija `trial`, `role`, `seed`, `protocol=1`,
y responde READY. STEP requiere `trial`, `sequence` empezando en 0 y
`time_ms` con incrementos de 12 ms. El primer tiempo puede ser cero o un
offset de simulación. Internamente se añade 100000 ms, como hace el reloj
original para no confundir cero con un evento inexistente.

Sensores: `angles[26]`, `velocity[26]`, `gyro[3]` en rad/s, `acc[3]` en m/s²
de fuerza específica (+g vertical en reposo), `orientation[4]` wxyz unitario
con yaw de mundo incluido, `fsr[8]` en kg equivalente. FSR se ordena izquierda
fl/fr/bl/br y derecha fl/fr/bl/br. Articulaciones siguen `HELLO.joint_names`
y el orden original Joints, incluyendo ambas manos y HipYawPitch derecho.

`request.motion` admite stand/walk/kick/block/getup. Marcha usa forward/lateral
en mm/s y turn en rad/s. Patada admite forwardFastLeft/forwardFastRight y
ball[2] en milímetros del plano robot. Block usa center/left/right; las fases
son genuflectStand, keeperJumpLeft y su espejo. Cabeza usa head_yaw/head_pitch
en radianes. Getup solicita stand: el motor original decide la recuperación
según FallDownState, sin forzar estados inventados.

OUTPUT devuelve ángulos, rigidez resuelta 0..100, odometría en mm/radianes,
nombre de fase original, petición, busy y pulso kick_completed. Off/ignore
se resuelven con la medida actual antes de enviar al servo. Nunca se
aplican valores no finitos. ERROR termina el proceso; reiniciar exige
crear uno nuevo. SHUTDOWN termina ordenadamente.
