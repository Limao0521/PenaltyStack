# Código B-Human del penal (overlay)

Todo lo que el NAO ejecuta en la prueba de penal vive aquí, en PenaltyStack.
No se hacen commits ni ramas en `SabanaHerons2026`.

- `files/`: archivos de B-Human que el penal agrega o modifica, con su ruta dentro del fork.
- `overlay.json`: commit base del fork (`0d1ea7ab`, en `origin/vision-new-dataset`) y lista de archivos (`added`/`modified`).
- `../tools/bhuman_overlay.py`: copia `files/` a un checkout limpio del commit base y compila.

## Flujo

```bash
cd ~/workspace/PenaltyStack
.venv/bin/python tools/bhuman_overlay.py worktree   # una vez: ~/workspace/semillero/SabanaHerons2026-penalty
.venv/bin/python tools/bhuman_overlay.py build      # aplica el overlay y compila Nao + DeployDialog
.venv/bin/python tools/bhuman_overlay.py status     # qué falta aplicar o choca
~/workspace/semillero/SabanaHerons2026-penalty/Make/Common/deployDialog Develop   # ventana de deploy
```

El worktree es un checkout separado del fork en el commit base; el checkout
principal `~/workspace/semillero/SabanaHerons2026`, con trabajo de otras
sesiones sin commit, nunca se toca. `apply` se niega a sobrescribir un archivo
que cambió fuera del overlay y restaura los que se quitan del overlay.

Para editar: cambiar el archivo en `bhuman/files/` y correr `build`; o
editarlo en el worktree y traerlo con
`tools/bhuman_overlay.py capture <ruta relativa al fork>`. Después, commit en PenaltyStack.

## Contenido

| Parte | Archivos |
|---|---|
| Escenarios y ubicación Copa | `Config/Scenarios/CopaPenaltyStriker`, `CopaPenaltyKeeper`, `Config/Locations/CopaPenalty` (balón 32,5 mm), `Config/teams.cfg` |
| Botones, silbato, voz, sentado sin salto | `SkillBehaviorControl.*`, `CopaPenaltyState.h`, `PenaltyKeeper.h`, `Dive.cpp`, `BehaviorParameters.h` |
| Balón | `YoloBallDetector.*`, `CopaBallModel.h`, modelos `trionda_copa_upper.onnx` y `trionda_copa_lower.onnx`; balón naranja oficial: `CopaOrangeBall.h`, `CopaOrangeBallConfig.h`, `copaOrangeBall.cfg` |
| Postes azules, portería, otros NAO | `Src/Modules/Perception/CopaPerceptors/`, `Src/Representations/Perception/CopaPercepts/` |
| Stream al visor | `CamfStreamServer.*`, `CameraStreamer.*`, `CopaCameraStreamer.*`; visor `../tools/copa_camera_viewer.py` |
| Despliegue | `Make/Common/deploy`, `cameraStreamConfig`, `openCameraViewer`, `Src/Apps/DeployDialog/SettingsArea.*` |
| Robustez que ya corría en el robot probado | `UnscentedKalmanFilter.h`, `InertialDataProvider.cpp`, `MotionEngine.cpp`, `LibLookActiveProvider.cpp`, `BallPerceptFilter.cpp`, `YoloBallBridge.*`, `FieldColorNormalizer.h` |

Las claves nuevas de `LOADS_PARAMETERS` deben estar en todos los `.cfg` del
módulo (si falta una, B-Human hace FAIL y el robot queda con ojos rojos):
`tests/test_bhuman_overlay.py` lo comprueba.
