# Estado operativo — 2026-10-07

## Objetivo

Controles interactivos de simulación y ejecución del motor B-Human original en MuJoCo para tirador y portero.

## Completado

- Runtime B-Human original compilado con MSVC 2022, 559 originales preservados y verificados por hash.
- MuJoCo con raíces libres, servos, IMU/FSR; movimiento y contacto de balón físicos.
- Panel y atajos: iniciar, pausa, paso, reset, repetir, ABORT, velocidad, modo y acciones manuales.
- Dos procesos por ensayo, JSONL acotado, timeouts y cierre comprobados; sin esperas bajo mutex del visor.
- FSM espera finalización de patada y conserva primera guardia del disparo.
- Suite completa: 37 pruebas pasan; incluye visor/panel y tres penales idénticos con semilla 0. Informe `docs/validation/simulation-2026-10-07.json`.

## Estado y siguiente acción

- Rama: `feat/bhuman-simulation`. Lanzar `.venv/Scripts/python.exe -m penalty.simulator`, escena pausada; pulsar Iniciar.
- Operación/compilación/limitaciones: `docs/SIMULACION.md`, `native/bhuman_motion/README.md`.
- Baseline funcional simulado: marcha 0.469 m/5 s, kick izquierdo con contacto, bloqueo central, penal completo sin caída.
- Patada derecha falla contacto inicial; saltos laterales caen; GetUp no recuperó salto izquierdo. Resultados explícitos, no ocultar como éxito físico.
- Hardware: sigue sin penal completo habilitado; paquete NAOqi separado, ejecutor de patada apagado. Orientación/percepción ideales, sin calibración real.
- Siguiente etapa: calibrar variantes/recuperación y diseñar host NAO/LoLA con validación progresiva si el usuario la autoriza. Esta entrega no modifica hardware ni fuente original.
