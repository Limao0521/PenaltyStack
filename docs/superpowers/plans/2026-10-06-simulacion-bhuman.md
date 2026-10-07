# Simulación B-Human Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Controles interactivos y ejecución de los motores originales B-Human con dinámica MuJoCo para dos NAO.

**Architecture:** Python opera escena, sesión y panel; un ejecutable C++ por robot recibe sensores y devuelve JointRequest. Las fuentes B-Human se fijan al baseline migrado; el adaptador y la selección de proveedores quedan separados.

**Tech Stack:** Python existente, MuJoCo 3.3.7, Tkinter, C++17 o la versión requerida por las fuentes fijadas, CMake y MSVC.

**Spec:** `docs/superpowers/specs/2026-10-06-simulacion-interactiva-bhuman-design.md`.

## Global Constraints

- Fuente del motor: `2360ab1100ad28f70e190c893f6b095651ff0bbc`.
- Ciclo motriz: 0,012 s; física: 0,002 s; presupuesto activo del penal: 28 s.
- Raíces libres, contacto físico de patada, una instancia motriz por NAO.
- Preservar hashes/licencias y señalar entradas sintéticas del simulador.
- La ventana abre pausada y permanece visible tras terminar.
- No incorporar SimRobot ni desplegar movimientos físicos al NAO en esta entrega.

## Review Focus

- Reinicio mientras llega una respuesta: descartar secuencias de otra sesión.
- Cierre con proceso bloqueado: timeout y recogida de procesos sin congelar interfaz.
- Rigidez cero y sentinelas B-Human: desactivar esfuerzo sin objetivos residuales.
- Transición KICK/RECOVER: esperar fase ejecutada, evitando cancelar o repetir patada.
- Cambio de velocidad y pausa: conservar reloj simulado y reproducibilidad.

## Task 1: Runtime original y migración compilable

**Files:** `tools/import_motion_engine.py`, `tools/build_motion_runtime.py`, `native/bhuman_motion/`, `baseline/`, `tests/test_motion_runtime.py`.

**Interfaces:** Ejecutable `bhuman_motion` con HELLO, INIT, STEP, OUTPUT, ERROR y SHUTDOWN por JSONL; STEP contiene sensores y petición, OUTPUT contiene ángulos, rigidez y fase. Manifiesto preserva revisión y hashes.

- [x] Crear prueba de arranque/HELLO y lectura de articulaciones originales; observar fallo por runtime ausente.
- [x] Corregir resolución de includes y migrar dependencias de compilación/enlazado con procedencia.
- [x] Crear target CMake y adaptador C++ con contexto global/blackboard y módulos originales.
- [x] Configurar/compilar; resolver errores mediante diagnóstico dirigido.
- [x] Ejecutar runtime con postura, marcha, patada y keyframes; registrar fases y salidas articulares no constantes.
- [x] Comprobar rechazo de mensajes inválidos y secuencias; registrar resultado en ledger.

## Task 2: Puente de dinámica y sensores

**Files:** `penalty/bhuman_bridge.py`, `penalty/nao_model.py`, `penalty/physical_sim.py`, `tests/test_physical_sim.py`.

**Interfaces:** `BHumanBridge.step(sensor_frame, request) -> output`; `PhysicalSimulator.reset()`, `step(requests)`, `observation()`, `snapshot()`, `close()`. Tiempo físico avanza 12 ms por step.

- [x] Crear pruebas de raíces libres, mapeo/rigidez y paso de 12 ms; observar fallo inicial.
- [x] Añadir IMU/FSR, unidades y mapeo de articulaciones, control limitado de servos y procesos con timeout.
- [x] Inicializar postura con pies sobre suelo y ejecutar salidas originales sin mover raíces ni inyectar patadas.
- [x] Verificar postura 10 s y marcha 5 s; diagnosticar modelo/sensores/ganancias si cae.
- [x] Verificar contacto de patada y bloqueo; registrar caídas, contactos y métricas físicas.
- [x] Ejecutar tests del puente incluyendo timeout, salida inválida y rigidez cero.

## Task 3: Sesión interactiva y conexión con FSM

**Files:** `penalty/sim_session.py`, `penalty/sim_panel.py`, `penalty/simulator.py`, `penalty/control.py`, `tests/test_sim_session.py`.

**Interfaces:** `SimulationSession.command(name, value=None)`, `tick()`, `snapshot()`, `close()`; comandos start/pause/reset/repeat/step/abort/speed/manual. El panel usa cola de comandos y copia de telemetría.

- [x] Crear pruebas de pausa/reinicio/repetición/ABORT y retención de acciones; observar fallos iniciales.
- [x] Implementar estados, asentamiento, relojes, FSM con confirmación motriz y modos manual/penal.
- [x] Implementar panel Tkinter, atajos y visor; mantener UI receptiva y cierre coordinado.
- [x] Conectar CLI GUI/headless, configuración de pie/lado/velocidad y trazas por ensayo.
- [x] Verificar sesión real y pausa de pared, un paso, repetición, abortar y cierre durante intercambio.
- [x] Ejecutar pruebas de sesión y penal completo con motor real.

## Task 4: Entrega y revisión

**Files:** `README.md`, `AGENTS.md`, `docs/ARQUITECTURA.md`, `docs/SIMULACION.md`, `.ai/HANDOFF.md`, pruebas existentes del simulador.

- [x] Adaptar contratos de pruebas existentes para dinámica física y ejecutar suite del proyecto.
- [x] Ejecutar tres repeticiones con misma semilla y comparar trazas/poses.
- [x] Revisar diff de procesos, protocolo y dependencias; resolver regresiones encontradas.
- [x] Documentar compilación, lanzamiento, controles, entradas sintéticas y resultados realmente medidos.
- [x] Actualizar ledger, handoff y commits; presentar comando de uso y limitaciones concretas.

## Ejecución autorizada

El usuario aprobó el diseño y ordenó «implémentalo» tras recibir la especificación. Se ejecuta en esta sesión, sin otra ronda de permisos, en la rama `feat/bhuman-simulation` del checkout solicitado. Las modificaciones previas del visor se conservan. Codex realiza integración/diagnóstico C++; local-worker se usa para tareas acotadas cuando responde. Si no responde, Codex continúa directamente.
