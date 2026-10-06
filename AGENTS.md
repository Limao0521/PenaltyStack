# PenaltyStack

## Mapa

- `penalty/control.py`: máquina de estados compartida por simulación y NAO.
- `penalty/vision.py`: detector local de pelota naranja.
- `penalty/simulator.py`, `penalty/nao_model.py`, `penalty/joint_env.py`, `sim/scene.xml` y `models/bhuman/`: cancha, modelo V6 B-Human y acciones articulares MuJoCo.
- `tools/build_bhuman_model.py`: regeneración de MJCF y mallas desde la fuente RSI3 fijada.
- `penalty/nao.py`: adaptador NAOqi local; ninguna dependencia de Wi-Fi durante el penal.
- `deploy/`: empaquetado y despliegue Ethernet con ruta aislada.
- `heritage/`: recursos estrictamente seleccionados de SabanaHerons2026 y procedencia.
- `baseline/`: módulo MotionControl B-Human completo y dependencias seleccionadas, sólo como referencia de comparación.

## Comandos

- `python -m unittest discover -s tests -v`
- `python -m penalty.simulator --headless`
- `python tools/build_bhuman_model.py`
- `python deploy/package.py --profile profiles/striker.example.json --output dist/striker.tar.gz`

## Restricciones

- Dos roles configurables por robot; START y ABORT físicos, límite activo de 28 s.
- No ejecutar movimientos físicos sin ensayos incrementales y calibración del NAO concreto.
- Las mallas, masas y colisiones vienen de B-Human, pero la adaptación MuJoCo no reproduce todo SimRobotCore3; aún no hay control de equilibrio ni validación física.
- No incorporar SimRobot, GameController, red de juego ni política RL.

Ver `docs/ARQUITECTURA.md` y `docs/DESPLIEGUE.md`.
