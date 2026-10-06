# PenaltyStack

Mini stack separado para penales NAO v6. Repositorio de origen: [SabanaHerons2026](https://github.com/juanulloa2050/SabanaHerons2026), revisión `2360ab1100ad28f70e190c893f6b095651ff0bbc`.

El controlador de tirador y portero comparte contratos de observación y comando con el simulador MuJoCo y el adaptador NAOqi. El simulador anterior SimRobot, el GameController y la comunicación de equipo no forman parte de este repositorio. El módulo completo `MotionControl` de B-Human se conserva como referencia de código en `baseline/bhuman/`; aún no se ejecuta aquí.

![Cancha de penales con dos NAO V6 basados en el modelo de B-Human](sim/preview.png)

![Detalle frontal del NAO V6 de B-Human en MuJoCo](sim/nao_detail.png)

This product includes software developed by B-Human (http://www.b-human.de). Ver los términos del [modelo 2025](models/bhuman/License.md) y del [baseline original](heritage/upstream/License.md).

## Estado

- **Simulación:** dos NAO V6 construidos desde `NaoV6H25.rsi3` de B-Human, con 50 mallas visuales distintas, 24 articulaciones por robot, masas/inercia y primitivas de contacto. El flujo de penales desplaza las raíces de forma cinemática y usa un impulso de patada ideal. `JointSpaceEnv` expone 23 objetivos articulares por NAO para futuros ensayos de RL; no tiene política, recompensa ni marcha estable.
- **Motor original:** los 54 archivos de `Src/Modules/MotionControl` y su selección de dependencias/configuración (266 archivos en total) están preservados con hashes para comparación. No se han compilado, ejecutado ni conectado al adaptador NAOqi o a MuJoCo.
- **NAO:** cámara RGB, botón START, botón ABORT, control de cabeza y marcha acotada tienen adaptador onboard. Por defecto el proceso es de observación. El ejecutor físico de patada aún no está conectado; el adaptador rechaza `kick_enabled`.
- **Despliegue:** paquete Python para NAOqi 2.8 por Ethernet en `/home/nao/PenaltyStack`, aislado del servicio y los archivos de B-Human. No hay borrado remoto.

## Inicio rápido en PC

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-sim.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m penalty.simulator --headless
.\.venv\Scripts\python.exe tools/build_bhuman_model.py  # regenerar MJCF y mallas desde RSI3
```

Leer [arquitectura](docs/ARQUITECTURA.md), [operación y despliegue](docs/DESPLIEGUE.md) y [criterios para NAO real](docs/PREPARACION_NAO.md).
Ver también [modelo V6 y licencia](models/bhuman/README.md) y [baseline motriz](baseline/README.md).
