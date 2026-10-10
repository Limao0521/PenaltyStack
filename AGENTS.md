# PenaltyStack

## Mapa

- `penalty/control.py`: máquina de estados compartida por simulación y NAO.
- `penalty/vision.py`: detector local de pelota naranja.
- `penalty/simulator.py`, `penalty/nao_model.py`, `penalty/joint_env.py`, `sim/scene.xml` y `models/bhuman/`: cancha, modelo V6 B-Human y acciones articulares MuJoCo.
- `tools/build_bhuman_model.py`: regeneración de MJCF y mallas desde la fuente RSI3 fijada.
- `native/bhuman_motion/`: host C++ original por NAO; `tools/build_motion_runtime.py` compila con MSVC 2022.
- `penalty/bhuman_bridge.py`, `physical_sim.py`, `sim_session.py`, `sim_panel.py`: IPC acotado, física/sensores, ciclo/controles y panel Tk.
- `penalty/nao.py`: adaptador NAOqi local; ninguna dependencia de Wi-Fi durante el penal.
- `deploy/`: empaquetado y despliegue Ethernet con ruta aislada.
- `heritage/`: recursos estrictamente seleccionados de SabanaHerons2026 y procedencia.
- `baseline/`: originales sin modificar usados por el runtime y como referencia para RL.
- `penalty/rl/`, `penalty/skills/`, `penalty/optimization/`: RL táctico (contratos, entornos Gymnasium sintético/B-Human, recompensas versionadas, skills acotadas, búsqueda; entrenamiento asíncrono en `async_ppo.py`/`actors.py`, duelo entre políticas en `duel.py`). Diseño en `docs/rl/`, estado en `docs/RL.md`, corridas y resultados en `docs/RESULTADOS_RL.md`.

## Comandos

- `python -m unittest discover -s tests -v`
- `python -m penalty.simulator --headless`
- `python tools/build_bhuman_model.py`
- `python tools/build_motion_runtime.py`
- `python -m unittest tests.test_sim_acceptance tests.test_sim_gui -v`
- `python -m unittest tests.test_rl_contracts tests.test_rl_envs tests.test_rl_bhuman -v`
- `python tools/evaluate_rl.py --role keeper --policy geometric --episodes 100 --seeds 0 1 2 3 4`
- `python tools/train_rl.py --hours 48 --workers 30` (PPO hasta convergencia, `--hours` es tope; B-Human con recolección asíncrona; supervisado con reinicio; `--output <dir> --resume` para continuar)
- `python tools/duel_rl.py --strikers scripted v2=models/rl/striker_v2 --keepers scripted v4=models/rl/keeper_v4:gate --episodes 200 --workers 30` (duelo en B-Human; `:gate` para porteros entrenados con `commit_after_shot`)
- `python tools/watch_rl.py --striker models/rl/striker_v2 --keeper scripted --shots 4 --live` (visor de MuJoCo; `--video <carpeta>` graba MP4)
- Modelos entrenados en `models/rl/` (ver `manifest.json`); reportes en `docs/validation/rl/`.
- `python deploy/package.py --profile profiles/striker.example.json --output dist/striker.tar.gz`

## Restricciones

- Dos roles configurables por robot; START y ABORT físicos, límite activo de 28 s.
- No ejecutar movimientos físicos sin ensayos incrementales y calibración del NAO concreto.
- El flujo usa raíces libres, servos y contactos físicos. No escribir trayectoria de raíz ni velocidad de balón para producir movimientos.
- Preservar los bytes de `baseline/bhuman` y RSI3: las adaptaciones de plataforma se generan bajo `native/bhuman_motion/build/generated`.
- No esperar IPC bajo el mutex del visor. Una cola llena/fallo de canal también debe recolectar todos los hijos.
- Orientación/percepción ideales y sensores sintéticos deben quedar explícitos en trazas; no afirmar equivalencia con hardware.
- No incorporar SimRobot, GameController ni red de juego.
- RL solo en la capa táctica/skills: B-Human sigue como controlador de bajo nivel. El backend `synthetic` es analítico y siempre etiquetado; nada aprendido allí vale sin reevaluar en `bhuman`.
- Nunca editar una recompensa, objetivo u observación publicada: crear una versión nueva (`*_v2`).

Ver `docs/ARQUITECTURA.md`, `docs/SIMULACION.md` y `docs/DESPLIEGUE.md`.
