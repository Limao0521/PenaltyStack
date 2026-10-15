# Área 1 — Auditoría para el despliegue en robots reales

Fecha: 2026-10-15. Alcance: PenaltyStack (rama `codex/portero-visual-y-reentrenamiento-rl`),
SabanaHerons2026 (`2360ab1`, base B-Human Code Release 2023) y el reglamento
*Copa NAO CCM MX 2026 V1.0*, categoría **3.3 Tiro Penal**.

## Qué existía en PenaltyStack

| Capa | Estado antes de este trabajo | Utilidad para el robot real |
| :--- | :--- | :--- |
| Simulación MuJoCo + motor B-Human original (`native/bhuman_motion`) | Completa: marcha, patadas `forwardFast`, seis defensas originales, levantarse. | Prueba que las políticas ya mandan **acciones B-Human**, no servos: se pueden ejecutar con el mismo motor en el NAO. |
| Políticas tácticas (`models/rl/`) | Tirador v2 recomendado; v4, v3 y v1 experimentales o anteriores. Portero visual v2 `balanced` de referencia. | Son pequeñas (MLP 19→128→128→4) y deterministas: exportables a C++ sin PyTorch. |
| Percepción simulada (`penalty/perception.py`) | Rastreador de balón con covarianza y velocidad; `vision_keeper_v2`. | Lógica pura, portable. |
| Despliegue NAOqi (`penalty/nao.py`, `deploy/package.py`) | Python 2 sobre NAOqi, **solo observación**: el adaptador rechaza `kick_enabled` y bloqueos. | No sirve para competir: NAOqi no ejecuta las patadas ni las defensas con las que se entrenaron las políticas. |
| Reglas de competencia | FSM con START/ABORT y 28 s; cancha de la simulación ya igual a la regla 3.3.4. | Base para el controlador del robot. |

Medidas de `sim/scene.xml` contra la regla 3.3.4: línea de gol a 1,30 m de la marca, tirador
20 cm detrás, portería de 1,50 m × 0,80 m, área chica de 0,60 m × 2,20 m, balón naranja
de 65 mm y 55 g. Todo coincide, así que las políticas se entrenaron en la geometría
de la competencia.

## Qué aportan B-Human 2023 y SabanaHerons2026 para el robot

- Runtime completo del NAO V6 (LoLA, cámaras V4L2, IMU, botones), con el mismo
  `MotionEngine`, `WalkingEngine`, `KickEngine` y `KeyframeMotionEngine` que la simulación.
  Incluye levantarse tras una caída.
- Las skills `WalkToBallAndKick` (patada con dirección y alcance), `Dive` (las seis defensas
  originales), `LookAtPoint`, `Stand` y `PlayDead`.
- `WhistleRecognizer` de tres perfiles de SabanaHerons (silbato de árbitro, de mano y de boca).
- Toolchain cruzado (clang + mold + sysroot V6 en `Util/Buildchain`), el servicio
  `bhuman.service` y los scripts `deploy`/`setupSSH` con la llave de los robots.
- La configuración por robot (`Config/Robots/<nombre>/network.cfg`, 192.168.49.x) y los
  perfiles de red, incluido `NONE`, que apaga el Wi-Fi.

## Brechas encontradas

1. **Sin GameController**: el penal de la Copa no lo usa. `GameStateProvider` alterna estados
   con el botón del pecho y exige 3 votos para el silbato (`minVotersForWhistle = 3`).
2. **Pelota y piso distintos**: B-Human y SabanaHerons detectan balón SPL o HSL sobre césped
   verde. La Copa usa balón naranja de plástico sobre duela con líneas negras. La
   autolocalización por líneas y `BallPerceptFilter` (que descarta balones "fuera del campo")
   no son fiables allí.
3. **Políticas en Python/PyTorch**: no corren en el NAO tal cual.
4. **Despliegue**: el `DeployDialog` de B-Human es Qt para partidos HSL (equipo, colores,
   modos RL de 4v4). No ofrece rol de penal, políticas de penal ni modo de arranque.
5. **Cámaras alternadas**: el hilo de cognición recibe cuadros de la cámara superior y de la
   inferior alternadamente. Un cuadro sin balón de la otra cámara reiniciaría la confirmación
   del portero.

## Matriz de cumplimiento del reglamento

| Regla | Exigencia | Implementación |
| :--- | :--- | :--- |
| 2.7 | Cargar programas solo por cable; prohibidas las redes inalámbricas | `deployCopaPenalty` usa la IP LAN y activa el perfil `NONE` (Wi-Fi apagado) por defecto. La ventana pide confirmación si se desmarca. |
| 2.7 | Activación por sensores ≤ 10 s tras el silbato | Toque del botón medio de la cabeza; silbato opcional (`whistleOnly` o `buttonOrWhistle`). |
| 2.7 / 3.3.6 | No arrancar antes del silbato (repetición; 3 = descalificación) | `waiting`: de pie e inmóvil, cabeza fija. El silbato se ignora 2 s tras armar. Por defecto solo arranca con botón (no hay falsos silbatos de otras categorías). |
| 2.7 | Operación 100 % autónoma | Sin red ni PC durante el penal; todo corre en el NAO. |
| 3.3.3 | Desactivación manual con un sensor que no esté en las piernas | Botón trasero de la cabeza mantenido 0,5 s → `aborted` (de pie, sin moverse). |
| 3.3.3 | Los tres robots deben tirar al menos una vez | Cada robot lleva ambas políticas. El rol cambia en el robot (botón delantero 2 s) sin redesplegar. |
| 3.3.5 | Ejecutar el tiro en 30 s tras el silbato | Presupuesto activo de 28 s; luego `finished`. |
| 3.3.5 | El portero solo se desplaza dentro del área chica | El portero no camina: guardia y defensas en el sitio. |
| 3.3.5 | Repetición si los robots se tocan | El tirador nunca persigue el balón; se detiene si se aleja más de 0,9 m de su salida. |
| 2.7 | Permanecer en el área hasta que el juez autorice | `finished`/`aborted`: de pie e inmóvil. Se rearma con el botón delantero para el siguiente tiro. |

## Riesgos que siguen abiertos

- **Sin prueba en hardware**: compilación y paridad sí; robot físico no. Las cifras de goles y
  atajadas son de simulación.
- **Umbrales de color** del balón (`copaOrangeBallPerceptor.cfg`): calculados para naranja
  frente a madera y piel; hay que medirlos con las cámaras en la sede.
- **Calibración de cámara y cinemática** por robot: viene de la configuración de SabanaHerons.
  Hay que verificarla, porque la proyección a suelo depende de ella.
- **Alcance de las estiradas** del portero: no calibrado en robot real. Las caídas laterales
  dañan hardware si se repiten; con `--jumps off` se usan guardias amplias sin caída.
