# PenaltyStack

Mini stack separado para penales NAO v6. Repositorio de origen: [SabanaHerons2026](https://github.com/juanulloa2050/SabanaHerons2026), revisión `2360ab1100ad28f70e190c893f6b095651ff0bbc`.

El controlador de tirador y portero comparte contratos de observación y comando con el simulador MuJoCo y el adaptador NAOqi. El simulador anterior SimRobot, el GameController, la comunicación de equipo y las políticas RL no forman parte de este repositorio.

![Vista de la cancha y dos NAO simplificados](sim/preview.png)

This product includes software developed by B-Human (http://www.b-human.de). Ver la licencia completa en `heritage/upstream/License.md`.

## Estado

- **Simulación:** modelo geométrico simplificado de dos NAO y balón físico en cancha de 2,4 × 2,4 m. Sirve para comprobar flujo y lógica; no representa equilibrio ni una patada NAO validada.
- **NAO:** cámara RGB, botón START, botón ABORT, control de cabeza y marcha acotada tienen adaptador onboard. Por defecto el proceso es de observación. El ejecutor físico de patada aún no está conectado; el adaptador rechaza `kick_enabled`.
- **Despliegue:** paquete Python para NAOqi 2.8 por Ethernet en `/home/nao/PenaltyStack`, aislado del servicio y los archivos de B-Human. No hay borrado remoto.

## Inicio rápido en PC

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-sim.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m penalty.simulator --headless
```

Leer [arquitectura](docs/ARQUITECTURA.md), [operación y despliegue](docs/DESPLIEGUE.md) y [criterios para NAO real](docs/PREPARACION_NAO.md).
