# Estado operativo — 2026-10-07

## Objetivo

Guardar integración y corregir patadas/recuperación lateral. El usuario diseñará RL después.

## Completado

- Runtime B-Human original compilado con MSVC 2022, 559 originales preservados y verificados por hash.
- MuJoCo con raíces libres, servos, IMU/FSR; movimiento y contacto de balón físicos.
- Panel y atajos: iniciar, pausa, paso, reset, repetir, ABORT, velocidad, modo y acciones manuales.
- Dos procesos por ensayo, JSONL acotado, timeouts y cierre comprobados; sin esperas bajo mutex del visor.
- FSM espera finalización de patada y conserva primera guardia del disparo.
- Perfil v2: orientación angle-axis original, radio real 32.5 mm, fase de golpe derecho 200 ms; endpoints, IK y balanceador originales.
- Ambos pies golpean durante kick una vez, con confirmación y sin caída; desplazamiento izquierdo 1.014 m, derecho 0.962 m a alcance pedido 2500 mm.
- Ambos saltos laterales recuperan automáticamente: GetUp → walk → stand, gate estable 0.5 s y un segundo adicional erguido con ambos pies apoyados. Caída histórica sigue enclavada; panel muestra postura actual.
- Suite completa: 45 pruebas pasan en 165.306 s; visor/panel y tres penales idénticos (blocked). 559 originales reverificados por hash. Informe `docs/validation/simulation-fixes-2026-10-07.json`; informe previo preservado.

## Estado y siguiente acción

- Rama: `feat/bhuman-simulation`. Lanzar `.venv/Scripts/python.exe -m penalty.simulator`, escena pausada; pulsar Iniciar.
- Operación/compilación/limitaciones: `docs/SIMULACION.md`, `native/bhuman_motion/README.md`.
- Baseline funcional simulado: marcha 0.469 m/5 s, ambos kicks con contacto, guardia central y saltos laterales con recuperación. Alcance pedido manual/CLI de 1500–3500 mm; verificado físicamente el caso inicial de 2500 mm, no todo el rango.
- Hardware: sigue sin penal completo habilitado; paquete NAOqi separado, ejecutor de patada apagado. Orientación/percepción ideales, sin calibración real.
- Siguiente etapa: recibir diseño RL del usuario y comparar contra el mismo mundo/perfil registrado. No hay política RL nueva ni despliegue físico en esta entrega. Integración anterior ya estaba limpia en `9c08a41`; las correcciones se guardan en el commit posterior.

## RL táctico P0 — 2026-10-07

- Diseño del usuario en `docs/rl/` (copiado del zip); estado y comandos en `docs/RL.md`.
- Hecho: contratos `rl_obs_v1`, entornos Gymnasium `synthetic`/`bhuman`, wrappers (historia, retardo, ruido/pérdida), recompensas v1, métricas, skills acotadas, búsqueda aleatoria/CMA-ES, `tools/evaluate_rl.py`. Suite: 79 pruebas.
- Hooks mínimos: `Command.block_now`, `SimulationSession.control_due`, `PhysicalSimulator.goal_crossing`.
- Hallazgos: girar en el sitio cerca del balón lo empuja → pose de patada orbitando; `kick_length_mm` 3100 desestabiliza la física → `power` mapea a 1500–2900 mm.
- Baseline B-Human (5 semillas × 4): tirador scripted 100 % bloqueado; portero geométrico vs `random_kicks` 45 % atajadas. Throughput ~0.15 ep/s por proceso, ~0.8 ep/s con 8.
- Siguiente: PPO (instalar stable-baselines3/torch), exponer parámetros de patada en runtime generado, self-play, adaptador NAO.

- 2026-10-07: PPO nocturno listo (`tools/train_rl.py`, `penalty/rl/training.py`); prueba corta de 9 etapas completada sin fallos. Lanzar `tools/train_rl.py --hours 10 --workers 8` y revisar `runs/train/<fecha>/report.json`.

## Entrenamiento RL — 2026-10-08 a 10

- Rama `feat/rl-contracts`, todo subido: código, `docs/RESULTADOS_RL.md`, modelos (`models/rl/`, con `manifest.json`), reportes y curvas (`docs/validation/rl/`) y videos (`docs/media/rl/`). Suite: 112 pruebas.
- Qué se hizo, resultados y dónde está cada cosa: `docs/RESULTADOS_RL.md`. Las copias completas, con todos los checkpoints y los logs de consola, solo están en este PC, en `runs/results/`.
- Tirador a usar: `models/rl/striker_v2`. Anota 0.96 contra el portero FSM con variación y ruido, y 91.5 % en el duelo; reparte los tiros 51/49 entre las dos esquinas.
- Portero: ningún entrenado supera la guardia central tras la patada (0.55). El mejor es el v4 (0.37); el geométrico, 0.18.
- Portero `anticipation` (`penalty/rl/anticipation.py`, predictor en `models/rl/kick_side_predictor.json`): lee la postura del tirador y ataja 24 % contra el tirador v2 (guardia central 4 %); con patadas aleatorias empata con la guardia central. El límite es la estirada de B-Human (0.46–0.59 incluso en el lado correcto).
- Portero recomendado en simulación frente a varios estilos: `robust` (`penalty/rl/robust_keeper.py`), configuración `models/rl/keeper_robust_v1.json`. Ocho estilos reservados: media 84.6 %, mínimo 77 %; con perturbaciones y 100 ms de retraso: 70.2 %, mínimo 57 %. Sin reentrenar PPO ni cambiar movimientos originales. El ensayo anterior de compromiso inmediato no era un techo: importa el tiempo de activación. Ver `docs/PORTERO_ROBUSTO.md` y `docs/validation/rl/robustez_20261010/`.
- PC de entrenamiento: solo VS Build Tools 2026 (el runtime se compila a mano con el generador "Visual Studio 18 2026"), Smart App Control desactivado y modo de energía "Máximo rendimiento".
- Siguiente: reducir los compromisos tempranos (26/800 con perturbaciones), evaluar otro controlador de tirador y calibrar percepción. Reconsiderar optimización de estirada tras fijar el tiempo de activación. Los cinco errores del test original corresponden a dos semillas del tirador con sensores fuera de rango; permanecen contados.
