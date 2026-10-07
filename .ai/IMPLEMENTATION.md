# Ledger — implementación B-Human/MuJoCo

- Autorización: diseño aprobado e implementación solicitada; rama `feat/bhuman-simulation` en el checkout PenaltyStack.
- Task 1 completa: 559 originales, runtime C++ MODULE/Blackboard, MSVC 2022 y JSONL; seis pruebas nativas verificadas.
- Task 2 completa: raíces libres, IMU/FSR/servo, ciclo 12 ms/2 ms y procesos por rol. Marcha sin caída y patada izquierda con contacto. La derecha termina su fase pero falla el contacto inicial.
- Task 3 completa: panel, atajos, modos, relojes, asentamiento, pausa/paso/reset/repeat/ABORT, FSM con confirmación de patada y trazas. Prueba conjunta del visor/panel pasa.
- Task 4 completa: suite de 37 pruebas pasa, tres repeticiones idénticas y documentación/evidencia final. Build final vuelve a compilar y seis pruebas nativas pasan después del build.
- Revisión independiente encontró cierre omitido si SHUTDOWN no cabe y mutex del visor durante IPC: corregidos. Prueba de cierre adverso pasa. Reloj activo congelado durante ABORT.
- Diagnóstico: liberar/cambiar bloqueo según velocidad/rumbo oscilaba entre guardia y recuperación; la primera guardia queda enclavada por ensayo. Penal completo vuelve a pasar sin caída.
- Rulings: preservar originales; generar adaptaciones de plataforma/diagnóstico fuera de baseline; ejecutar en checkout autorizado sin más aprobaciones.
- Limitaciones: saltos laterales al suelo; GetUp entra en fase pero no recuperó salto izquierdo. Sin calibración ni ejecutor nuevo para NAO real. Registrar fallos como fallos físicos.
- Local-worker GUI agotó 900 s sin archivo; Codex tomó la implementación. Revisión local posterior rechazada por detectar nuestra compilación concurrente fuera de su scope; no se acepta como revisión exitosa. Revisor independiente dejó dos hallazgos antes de agotar disponibilidad; Codex inspeccionó el cierre y verificó regresiones.
