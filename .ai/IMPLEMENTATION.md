# Ledger — implementación B-Human/MuJoCo

- Autorización: diseño aprobado e implementación solicitada; rama `feat/bhuman-simulation` en el checkout PenaltyStack.
- Task 1 completa: 559 originales, runtime C++ MODULE/Blackboard, MSVC 2022 y JSONL; seis pruebas nativas verificadas.
- Task 2 completa: raíces libres, IMU/FSR/servo, ciclo 12 ms/2 ms y procesos por rol. Marcha sin caída y ambas patadas con contacto físico tras las correcciones del perfil v2.
- Task 3 completa: panel, atajos, modos, relojes, asentamiento, pausa/paso/reset/repeat/ABORT, FSM con confirmación de patada y trazas. Prueba conjunta del visor/panel pasa.
- Task 4 de entrega anterior: suite de 37 pruebas pasó, tres repeticiones idénticas y documentación/evidencia histórica. Build de esa entrega compiló y sus seis pruebas nativas pasaron. Ver validación actual abajo.
- Revisión independiente encontró cierre omitido si SHUTDOWN no cabe y mutex del visor durante IPC: corregidos. Prueba de cierre adverso pasa. Reloj activo congelado durante ABORT.
- Diagnóstico: liberar/cambiar bloqueo según velocidad/rumbo oscilaba entre guardia y recuperación; la primera guardia queda enclavada por ensayo. Penal completo vuelve a pasar sin caída.
- Rulings: preservar originales; generar adaptaciones de plataforma/diagnóstico fuera de baseline; ejecutar en checkout autorizado sin más aprobaciones.
- Correcciones solicitadas el 7 de octubre: orientación angle-axis original; radio real 32.5 mm; fase derecha 3 de 200 ms con endpoints intactos. Alcance pedido 1500–3500 mm por panel/CLI. Saltos laterales liberan dive tras entrar en fase original y recuperan automáticamente; guardia central permanece enclavada. Panel distingue postura actual/caída histórica.
- Validación actual: 45 pruebas pasan en 165.306 s. Ambas patadas con contacto durante kick y sin caída; ambas recuperaciones alcanzan stand, estabilidad 0.5 s y apoyo/torso erguidos un segundo adicional. Tres penales idénticos con resultado blocked. 559 originales verificados por hash. Informe `docs/validation/simulation-fixes-2026-10-07.json`.
- El informe de 37 pruebas conserva los fallos anteriores como evidencia histórica; no describe el perfil actual. Sin calibración ni ejecutor nuevo para NAO real. RL permanece pendiente del diseño del usuario.
- Local-worker GUI agotó 900 s sin archivo; Codex tomó la implementación. Revisión local posterior rechazada por detectar nuestra compilación concurrente fuera de su scope; no se acepta como revisión exitosa. Revisor independiente dejó dos hallazgos antes de agotar disponibilidad; Codex inspeccionó el cierre y verificó regresiones.
