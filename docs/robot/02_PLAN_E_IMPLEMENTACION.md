# Área 2 — Planeación e implementación en el framework del robot

## Decisión

El robot ejecuta **B-Human 2023 / SabanaHerons2026 completo**, con un escenario propio
`CopaNAOPenalty`. No se usa NAOqi. Así las políticas mandan las mismas primitivas con las que
se entrenaron en MuJoCo: `forwardFastLeft/Right` con alcance, las seis defensas
`MotionRequest::Dive` y la marcha. El riesgo físico queda en el motor que el equipo ya usa en
competencia.

## Qué se copió a este repositorio

`robot/` contiene el subconjunto de SabanaHerons2026 `2360ab1` necesario para compilar y
desplegar el binario `bhuman` del NAO V6. Son 4994 archivos (148 MB):

- `CMakeLists.txt`, `Make/{CMake,Common,Linux}`, `Src/`, `Config/` (sin escenas de SimRobot)
  e `Install/` (servicio, perfiles de red, llave SSH de los robots);
- `Util/Buildchain/V6` (sysroot) y las librerías V6/Linux: fftw, flite, hdf5, libjpeg,
  onnxruntime, etc.;
- los submódulos `Util/CompiledNN` (`091512e`) y `Util/SimRobot` (solo Eigen, `732794c`).

Lo importa `tools/import_robot_framework.py` de forma reproducible. `robot/PROVENANCE.json`
guarda el SHA-256 de cada archivo original. Para listar qué cambió PenaltyStack:

```bash
python tools/robot_build.py --check-originals
```

Hay **15 originales modificados**: `SkillBehaviorControl.h` (dos `REQUIRES` y la nueva
opción) y una línea de `defaultRepresentations` en cada uno de los 14 `threads.cfg`. Todo lo
demás son **26 archivos nuevos** (`python tools/robot_build.py --check-originals` los lista).

## Arquitectura en el robot

```
Upper/Lower:  CameraImage ──► CopaOrangeBallPerceptor ──► BallPercept
Cognition:    EnhancedKeyStates + Whistle ──► CopaPenaltyController ──► CopaPenaltyState + GameState
              BallPercept + odometría + IMU ──► CopaPenaltyTactics ──► CopaPenaltyDecision
              SkillBehaviorControl / HandleCopaNAOPenalty ──► MotionRequest + HeadMotionRequest
Motion:       MotionEngine (Walking/Kick/Keyframe/GetUp originales de B-Human)
```

| Pieza | Archivo | Función |
| :--- | :--- | :--- |
| Controlador | `Src/Modules/BehaviorControl/CopaPenaltyController` | Fases `unstiff → waiting → playing → finished/aborted`, gestos de botones, silbato y presupuesto de 28 s. Llena `GameState` como un penal de B-Human (escucha de silbato y LEDs). |
| Táctica | `Src/Modules/BehaviorControl/CopaPenaltyTactics` | Rastrea el balón, reconstruye la pose desde la salida conocida, consulta la política del tirador cada 100 ms (igual que la FSM simulada) y decide la defensa del portero. |
| Núcleo portable | `Src/Tools/CopaPenalty/CopaPolicies.h` | C++ sin dependencias de B-Human: `rl_obs_v1`, actor PPO determinista, `BallTracker`, `vision_keeper_v2` y modos `balanced`/`original`. |
| Ejecución | `SkillBehaviorControl/Options/HandleCopaNAOPenalty.h` | Ejecuta la decisión con las skills originales: `WalkToPose`, `WalkToBallAndKick` (precisa, sin patada girada), `Dive`, `LookAtPoint` y `Stand`. |
| Balón | `Src/Modules/Perception/BallPerceptors/CopaOrangeBallPerceptor` | Segmenta el naranja en YUYV, excluye el propio cuerpo, compara el radio medido con el esperado de una pelota de 65 mm y proyecta al suelo con covarianza. |
| Escenario | `Config/Scenarios/CopaNAOPenalty`, `Config/Locations/CopaNAO_CCM2026` | Igual a `Default` salvo tres cosas: el balón, `GameState` y los proveedores nuevos. Sin YOLO, sin PPO de partido; `keeperJumpingOn = true`; balón de 32,5 mm de radio. |
| Políticas | `Config/NeuralNets/CopaPenalty/*.cfg`, `policies.json` | Pesos exportados por `tools/export_robot_policies.py`, con su hash, estado y descripción. |

### Comportamiento según el reglamento

- **Encendido**: articulaciones sueltas.
- **Pecho**: se pone de pie y espera inmóvil; dice "striker ready" o "keeper ready".
- **Botón medio** (o silbato, si está habilitado): empieza la rutina.
- **Tirador**:
  - Busca el balón y camina a la pose detrás de él, orientada al punto de tiro de la política.
  - Cuando el disparador de la política vale ≥ 0,5 y el balón está a ≤ 40 cm, libera **una
    sola** patada. B-Human se alinea y patea.
  - Luego se queda quieto 1,5 s y termina. Nunca sigue el balón.
- **Portero**:
  - De pie, sigue el balón con la cabeza.
  - `vision_keeper_v2` confirma el tiro con dos cuadros entrantes y decide el lado.
  - Ejecuta la defensa original a la hora decidida, la mantiene 3 s y B-Human lo levanta.
- **Fin**: a los 28 s, o cuando el tirador termina, el robot queda de pie e inmóvil. El botón
  delantero lo rearma para el siguiente penal.

### Decisiones de ingeniería

- **Balón propio**: la táctica rastrea `BallPercept` directamente y entrega esa estimación al
  motor de patada. Así no depende de la autolocalización sobre madera.
- **Cámaras alternadas**: un cuadro sin balón de la otra cámara se ignora durante 100 ms
  (`otherCameraGraceMs`); si no, el portero nunca confirmaría el tiro.
- **Rival en la observación del tirador**: se usa la pose nominal del portero sobre la línea
  de gol, como en el entrenamiento (información de escena con ruido).
- **Cambio de rol en el robot**: los tres robots deben tirar; cada uno lleva el tirador y el
  portero elegidos en el despliegue.

## Verificación hecha

| Comprobación | Resultado |
| :--- | :--- |
| Compilación cruzada NAO V6 (Release, clang 18 + mold, `-Wall -Wextra -Wconversion`) | Correcta, 0 advertencias en el código nuevo. |
| Paridad C++ ↔ Python (`tests/test_robot_policy_parity.py`) | **35 686 comprobaciones, 0 fallos**: 4 tiradores × 64 pasos, 72 observaciones, 24 tiros con rastreador y portero (`balanced`/`original`). |
| Pesos exportados frente a los publicados | Idénticos en float32. Hashes en `policies.json`. |
| Grafo de módulos (`tools/check_module_graph.py`) | `CopaNAOPenalty`: 136 módulos, 0 errores. `Default` y `4v4_Complete` siguen válidos. Detecta proveedores faltantes. |
| Simulacro de despliegue en WSL (`tests/test_robot_deploy.py`) | Configuración por robot correcta, Wi-Fi `NONE`, manifiesto y hash. |

**No verificado**: ejecución en un NAO físico, umbrales de color con la luz de la sede y
tiempos reales de patada y defensa. Los resultados de goles y atajadas son de simulación.
