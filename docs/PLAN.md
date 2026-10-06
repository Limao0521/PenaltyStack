# Extracción y entrega

1. Crear repositorio separado y registrar procedencia del origen.
2. Implementar controlador pequeño con límite, START, ABORT, una sola patada y ambos roles.
3. Crear cancha MuJoCo reglamentaria aproximada y conectar el controlador al adaptador.
4. Conectar cámara y sensores locales NAOqi para modo de observación; empaquetar y desplegar en ruta aislada.
5. Verificar tests, simulación y paquete. La habilitación de movimiento real exige pruebas con hardware.

El motor B-Human original no es una dependencia oculta del nuevo runtime: sus keyframes y parámetros se conservan como material de migración para la futura integración motriz. La diferencia queda explícita en la evaluación de preparación.
