# Estado operativo — 2026-10-10

## Actualización: seis defensas originales y biblioteca v6 (2026-10-10)

- Rama `codex/mejora-portero`. Host recompilado: `center`, `center_right`, `wide_left`, `wide_right`, `left`, `right`; `prepare` separado. HELLO anuncia `keeper_defenses_v2` sin romper protocolo 1. Panel manual acepta las siete solicitudes; el simulador automático general conserva la FSM anterior.
- Nuevo `keeper_action_v2` con nombre de defensa; contratos, objetivos y recompensas publicados intactos. `VisionLibraryKeeper` (`vision_keeper_v6`, experimental) compone v2: idéntica trayectoria, confirmación y activación. `balanced` elige centro según signo del cruce y conserva saltos; elige cuatro defensas. Las anchas se evalúan en modos separados. No hay entrenamiento nuevo ni modificaciones de keyframes.
- Predictor: RGB y sensores propios, dos capturas entrantes, mediana hasta tres cruces lineales a x=-0.12 m. Centro entre ±0.30 m se activa al confirmar; saltos con llegada ≤1.8 s. Primera defensa fija y límite 28 s. Yaw del rival se mide pero v2/v6 no lo usa; extrapola balón ya en movimiento, no intención previa.
- Suite: 221 pruebas pasan (85.311 s). MuJoCo: 7/7 solicitudes ejecutadas y recuperadas con apoyo estable adicional 1 s, hijos recogidos; una postura/semilla, sin tiro. Traza: 87/87 acciones exactas y cero diferencias de lado/activación frente al adaptador publicado v2.
- Integración: 80 ejecuciones, ocho estilos por dos semillas, cinco variantes pareadas, cero `sim_error`. Por 16 casos: original 8 atajadas; centro espejado 11; ancho izquierdo 8; ancho derecho 9; `balanced` 11. Muestra preliminar, sólo dos clusters de semilla: no usar el IC bootstrap para declarar superioridad. Salto derecho validado por separado, no seleccionado por estos casos.
- Evidencia congelada y hashes: `docs/validation/vision/keeper_defenses_v6`; explicación `docs/DEFENSAS_PORTERO.md`. CLI: `tools/watch_keeper_defenses.py --mode balanced --style v2 --live`, `tools/validate_keeper_defenses.py`, `tools/evaluate_keeper_defenses.py`. GIF regenerables en `runs/keeper_defenses/motions`; PNG y trazas en el reporte.
- Sólo simulación: sensores ideales y reloj físico pausado durante inferencia explícitos. 559 originales B-Human y RSI3 preservados. No habilitar movimiento físico sin calibración y ensayos incrementales; no afirmar latencia/rendimiento de hardware.

## Actualización: revisión de visión, portero v4/v5 (2026-10-11)

- v3 no usaba la orientación: exige postura ≤0.5 s antes del compromiso; en 200 penales de desarrollo la última postura válida estaba 0.7–2.8 s antes. La línea del torso predice el cruce con ~0.18 m de error (depende del pie); el balón da ~3 cm con dos capturas.
- Nuevo `penalty/keeper_fusion.py`: `pose_v4` (v2 + postura congelada al dejar el balón su reposo, fusión calibrada y retención de centro contradicho) y `full_v5`/`ball_v5` (mínimos cuadrados, centro a 1.8 s, límites 0.35 m izq./0.30 m der.). Modelo `models/vision/kick_heading_v1.json` ajustado en semillas 69001.
- Hallazgo de movimiento: la guardia central `squatArmsBackLeft` es asimétrica y falla al lado derecho si se pide temprano; existe `squatArmsBackRight` (no usado: requiere recompilar el host).
- Test final Windows (semillas 70001, 50 por estilo, pareado): FSM base 35.2 %, v2 74.8 %, v3 75.0 % (postura usada 0/400), v4 75.0 % (1 decisión cambiada), v5 balón 73.5 %, v5 73.8 %. v2−FSM +39.5 pp (IC95 31.8–47.2); v4−v2 y v5−v5balón +0.2 pp; v5balón−v2 −1.2 pp (IC95 −5.8 a +3.0): el centro tardío pierde tiros centrales y gana en la frontera derecha. Recomendación: mantener v2. Resultados en `docs/validation/vision/fusion_final_20261011`; detalle en `docs/PORTERO_FUSION.md`. Para repetir sólo el duelo: `run_keeper_fusion_step2.bat`. Cambios sin commit.

## Actualización: orientación visual y sensores propios

- Rama `codex/mejora-portero`. Nueva ruta `keeper_sensor_camera_v2`: B-Human aporta cajas originales; `nao_visual_yaw_v1` compara RGB con 2294 referencias NAO para estimar yaw corporal. Referencias/manifest/posturas reproducibles en `models/vision/nao_yaw_v1`. Dominio simulado únicamente, no modelo de hardware.
- `sim_sensors.py`: lee jointpos, gyro, accelerometer y FSR propios; integra gyro/gravedad, odometría de apoyo y FK local para cámaras. No consulta poses globales del robot/rival/cámara en percepción. El motor nativo conserva el perfil propio ideal `mujoco-truth-v2`. Sensores ideales, aproximaciones/drift y reloj de inferencia quedan explícitos.
- Cabeza encuadra balón y cuerpo antes del tiro; vuelve al balón entrante/cercano. Nuevo estado v2 y RL v3 (32 campos); contratos y políticas publicados anteriores intactos. `vision_keeper_v3` acepta postura reciente de baja incertidumbre sólo en frontera de guardia; dos capturas de balón siguen confirmando; primera guardia no se cambia.
- Probe: 12 giros/144 capturas, 108 válidas, error mediano 1.14°, p95 3.41°, cero frente/espalda. 24 duelos/ocho estilos: 1533/1989 válidas, mediana 1.66°, p95 5.12°, cero frente/espalda/errores/compromisos tempranos. V3 y v2 5/8 atajadas, FSM base 2/8; postura no cambió decisiones en esta muestra. No afirmar mejora de atajadas por orientación.
- Comando: `.venv/Scripts/python.exe tools/watch_keeper_orientation.py --style v2 --seed 68003 --output runs/keeper_sensor_camera` (opcional `--live`). Reproducción exacta 71/71 decisiones; visor no probado manualmente. Comparación `tools/evaluate_keeper_vision.py --opponent-detector bhuman --orientation visual`. Suite final: 183 pruebas pasan. Informe `docs/ORIENTACION_PORTERO.md`, trazas/resultados `docs/validation/vision/orientation_20261010`.
- NAOqi permanece sólo observación con yaw desconocido. Siguiente hardware: datos/calibración/camisetas/iluminación reales y runtime Linux; no habilitar movimientos. Los 559 originales, 19 originales de detector y RSI3 preservados.

## Actualización: percepción del rival

- Rama `codex/mejora-portero`. Detector B-Human activo en proceso local de imágenes: red ONNX superior y contraste/clustering inferior, cuerpos originales compilados. Misma revisión fijada que movimientos; 19 originales/pesos/licencias importados aparte en `heritage/bhuman_robot_perception`, hashes verificados. Los 559 originales y RSI3 intactos.
- `tools/build_robot_perception.py` compila Windows/MSVC 2022 o 2026; CMake tiene destino Linux pendiente de compilar/probar para NAO. Protocolo `bhuman_robot_pixels_v1`, sin actuadores/red de juego; cierre acotado y recolección del hijo incluso con fallos/colas llenas.
- `opponent_perception.py`: proyección de pies, cercanía al balón, odometría, edad/covarianza; contraste requiere dos capturas. Camiseta, caída y yaw desconocidos; confianza inferior no medida. No inferir yaw de velocidad. `vision_keeper_v2` sigue decidiendo por balón y los contratos/recompensas publicados no se alteraron.
- `sim_opponent_vision.py`: conserva balón/cabeza inferiores publicados; ambas imágenes de cabeza alimentan el detector. Prueba de 30 cuadros: posición validada en 28, error mediano 11.4 cm frente al centro de pies; ensayo de 16 episodios/ocho estilos: 1080/1142 cuadros válidos, error mediano 22.8 cm, cero errores. Solo integración preliminar, sensores propios ideales y sin equivalencia con hardware. Fuentes/trazas/cámaras/reportes en `docs/validation/vision/bhuman_opponent_20261010`.
- NAOqi integrado en observación con `profiles/keeper.bhuman-vision.example.json`: START/ABORT/28 s, dual camera RGB, sin movimientos. Requiere calibración y host Linux x86-64 compatible; paquete exige `--vision-runtime` y `.so` fijada, rechaza Windows y movimientos habilitados. No hubo prueba de hardware. Ver `docs/VISION_BHUMAN_RIVAL.md`.
- Suite completa: 168 pruebas en 66.0 s; 21 focalizadas tras añadir verificación del cierre NAOqi con fallos de cámaras/log. No cambiar resultados históricos de 74/80 o 84.6 %; no hay mejora de atajadas atribuible a orientación en esta entrega.

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
- Comparación directa solicitada contra el base `scripted` (FSM del proyecto con movimientos B-Human): mismas 800 semillas/estilos de la prueba original, 84.6 % del nuevo frente a 45.8 % del base; contra v2, 90/100 frente a 3/100; al centro, ambos 89/100. No se ejecuta la pila de comportamiento upstream completa. Reportes adicionales `bhuman_base_realista.json` y `comparacion_bhuman_base.json`; política y física sin cambios.
- Percepción y ejecución real pendientes, detalladas en `docs/PORTERO_ROBUSTO.md`: ya hay detector RGB de pelota y lectura NAOqi, pero falta seguimiento con velocidad/tiempos, postura del tirador, adaptador de observaciones para `robust` y ejecutor de bloqueos físicos calibrado. Prioridad propuesta: balón y tiempo de llegada; postura como pista adicional. El 84.6 % solo corresponde a simulación. El caso `center` reproduce las decisiones del tirador base; se probaron también v1/v2 y cinco estilos adicionales.
- Visión integrada (primera etapa) en `docs/VISION_PORTERO.md`: `perception.py` proyecta RGB calibrado y sigue balón con odometría/tiempos/covarianza; `vision_state_v1`, `rl_obs_v2` y `vision_keeper_v2` (alias `vision`) deciden sin yaw del rival. Cámara en cabeza de MuJoCo, seguimiento de cabeza solo desde píxeles, bloques originales. Ensayo preliminar reservado: 74/80 frente a 44/80 base, ocho estilos, 160 episodios, semillas 65001000000–65001000009, cero errores/compromisos tempranos; sensores y calibración ideales. Reportes/trazas/fuentes `docs/validation/rl/vision_20261010/`. NAOqi observa y registra con START/ABORT/28 s; perfil nuevo rechaza movimientos/bloqueos y necesita calibración. `BHumanVisionAdapter` consume snapshots locales y selecciona tirador; los perceptores upstream/exportación real siguen pendientes. Sin orientación inferida; nunca convertir velocidad del rival en yaw. Suite completa 147 pruebas en 66.2 s, 33 focalizadas tras finalizar adaptador/cierre; 559 originales y RSI3 preservados. El anterior `robust`/contratos/resultados v1 siguen disponibles.
- PC de entrenamiento: solo VS Build Tools 2026 (el runtime se compila a mano con el generador "Visual Studio 18 2026"), Smart App Control desactivado y modo de energía "Máximo rendimiento".
- Siguiente: reducir los compromisos tempranos (26/800 con perturbaciones), evaluar otro controlador de tirador y calibrar percepción. Reconsiderar optimización de estirada tras fijar el tiempo de activación. Los cinco errores del test original corresponden a dos semillas del tirador con sensores fuera de rango; permanecen contados.
