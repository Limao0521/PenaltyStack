# Preparación para prueba física

**Estado actual: no listo para un penal funcional en NAO real.** Sí hay controlador y simulación ejecutables, y un paquete de observación onboard. Falta un ejecutor de patada compatible con la pelota y verificación física.

Puertas concretas para declarar listo:

1. Confirmar en cada NAO v6 que Python 2.7, `qi`, cámara, ALMemory y ALMotion funcionan con el paquete. Medir CPU/FPS y comparar la imagen RGB con color real de la pelota.
2. Calibrar cámara y distancia/bearing con imágenes etiquetadas de la pelota naranja de 65 mm en la cancha. Medir recall, falsos positivos y error de posición. Añadir detección de postes azules o comprobar una orientación inicial repetible.
3. Integrar el `KickEngine`/keyframes de B-Human en un runtime real, o validar una trayectoria NAOqi propia. Probarla primero con robot sujeto y luego con pelota; comprobar una sola patada, recuperación y ABORT.
4. Verificar marcha, equilibrio, zona de seguridad y botón ABORT con ambos roles. Ensayar el peor caso de timeout y fallo de cámara. El portero todavía solo bloquea estático.
5. Ejecutar un penal completo sin Wi-Fi/PC, medir tiempos (<30 s y objetivo normal <15 s), trayectoria y contactos, y repetir por cada robot/perfil. Solo entonces activar movimiento físico para competencia.

La simulación no sustituye estas medidas: sus NAO son proxies cinemáticos y el impulso de kick no representa el pie real.
