# PenaltyStack

## Mapa

- `penalty/control.py`: máquina de estados compartida por simulación y NAO.
- `penalty/vision.py`: detector local de pelota naranja.
- `penalty/simulator.py` y `sim/scene.xml`: adaptador y cancha MuJoCo.
- `penalty/nao.py`: adaptador NAOqi local; ninguna dependencia de Wi-Fi durante el penal.
- `deploy/`: empaquetado y despliegue Ethernet con ruta aislada.
- `heritage/`: recursos estrictamente seleccionados de SabanaHerons2026 y procedencia.

## Comandos

- `python -m unittest discover -s tests -v`
- `python -m penalty.simulator --headless`
- `python deploy/package.py --profile profiles/striker.example.json --output dist/striker.tar.gz`

## Restricciones

- Dos roles configurables por robot; START y ABORT físicos, límite activo de 28 s.
- No ejecutar movimientos físicos sin ensayos incrementales y calibración del NAO concreto.
- El modelo MuJoCo es una aproximación geométrica, no una validación de dinámica NAO.
- No incorporar SimRobot, GameController, red de juego ni política RL.

Ver `docs/ARQUITECTURA.md` y `docs/DESPLIEGUE.md`.
