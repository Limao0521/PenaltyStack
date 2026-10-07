# Preparación para prueba física

**Estado actual: no listo para un penal funcional en NAO real.** Sí hay controlador y simulación ejecutables, y un paquete de observación onboard. Falta un ejecutor de patada compatible con la pelota y verificación física.

Puertas concretas para declarar listo:

1. Confirmar en cada NAO v6 que Python 2.7, `qi`, cámara, ALMemory y ALMotion funcionan con el paquete. Medir CPU/FPS y comparar la imagen RGB con color real de la pelota.
2. Calibrar cámara y distancia/bearing con imágenes etiquetadas de la pelota naranja de 65 mm en la cancha. Medir recall, falsos positivos y error de posición. Añadir detección de postes azules o comprobar una orientación inicial repetible.
3. Llevar el runtime motriz a un host compatible con el NAO y su interfaz de actuadores, o validar una trayectoria NAOqi propia. KickEngine ya se ejecuta en MuJoCo mediante un binario Windows; ese binario no se puede desplegar en el NAO. Probar primero con robot sujeto y luego con pelota; comprobar una sola patada, recuperación y ABORT.
4. Verificar marcha, equilibrio, zona de seguridad y botón ABORT con ambos roles. Ensayar timeout y fallo de cámara. El portero de hardware aún mantiene postura estática; MuJoCo sí ejecuta los bloqueos originales, con caída en los saltos laterales.
5. Ejecutar un penal completo sin Wi-Fi/PC, medir tiempos (<30 s y objetivo normal <15 s), trayectoria y contactos, y repetir por cada robot/perfil. Solo entonces activar movimiento físico para competencia.

La simulación ejecuta el motor original sobre raíces libres, sensores físicos y contacto real pie–balón. Esto demuestra integración simulada. La orientación y percepción son ideales, las ganancias/contactos no están calibrados con la unidad real y la patada derecha todavía falla el contacto en el escenario inicial. Ver [validación simulada](SIMULACION.md); no usar esos resultados para habilitar movimiento en hardware.
