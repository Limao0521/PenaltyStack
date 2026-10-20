# Preparación para prueba física

> **Robots reales en la Copa NAO CCM MX 2026:** la ruta de despliegue correcta es el framework
> B-Human de `robot/` con `DeployCopaNAO.bat`; seguir la
> [guía del robot físico](GUIA_DESPLIEGUE_ROBOT_FISICO.md) y
> [docs/robot/03 · Despliegue y operación](robot/03_DESPLIEGUE_Y_OPERACION.md).
> Lo que sigue describe la ruta anterior (adaptador NAOqi Python 2, solo de observación,
> como el de [DESPLIEGUE.md](DESPLIEGUE.md)) y se conserva como contexto histórico.

**Estado actual de esta ruta legacy: no lista para un penal funcional en NAO real.** Sí hay controlador y simulación ejecutables, y un paquete de observación onboard. Falta un ejecutor de patada compatible con la pelota y verificación física.

Puertas concretas para declarar listo:

1. Confirmar en cada NAO v6 que Python 2.7, `qi`, cámara, ALMemory y ALMotion funcionan con el paquete. Medir CPU/FPS y comparar la imagen RGB con color real de la pelota.
2. Calibrar cámara y distancia/bearing con imágenes etiquetadas de la pelota naranja de 65 mm en la cancha. Medir recall, falsos positivos y error de posición. Añadir detección de postes azules o comprobar una orientación inicial repetible.
3. Llevar el runtime motriz a un host compatible con el NAO y su interfaz de actuadores, o validar una trayectoria NAOqi propia. KickEngine ya se ejecuta en MuJoCo mediante un binario Windows; ese binario no se puede desplegar en el NAO. Probar primero con robot sujeto y luego con pelota; comprobar una sola patada, recuperación y ABORT.
4. Verificar marcha, equilibrio, zona de seguridad y botón ABORT con ambos roles. Ensayar timeout y fallo de cámara. El portero de hardware aún mantiene postura estática; MuJoCo sí ejecuta los bloqueos originales y recuperación automática tras ambos saltos laterales.
5. Ejecutar un penal completo sin Wi-Fi/PC, medir tiempos (<30 s y objetivo normal <15 s), trayectoria y contactos, y repetir por cada robot/perfil. Solo entonces activar movimiento físico para competencia.

La simulación ejecuta el motor original sobre raíces libres, sensores físicos y contacto real pie–balón. Ambas patadas y ambas recuperaciones laterales pasan sus pruebas físicas simuladas. La orientación y percepción son ideales y las ganancias/contactos no están calibrados con la unidad real. La duración de golpe derecho de 200 ms es una calibración de este modelo, que requiere validación aparte en el robot. Ver [validación simulada](SIMULACION.md); no usar esos resultados para habilitar movimiento en hardware.
