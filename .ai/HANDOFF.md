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
