# Empaquetado, despliegue y operación

## Antes de empaquetar

Crear perfiles separados por NAO a partir de `profiles/*.example.json`. Completar `robot_id`, `role`, cámara y rumbo inicial medido. El paquete rechaza los marcadores `TO_FILL` y una patada física habilitada. Python 2.7 con `qi` debe estar disponible en NAOqi 2.8; comprobarlo en cada unidad. El proceso usa solo `127.0.0.1:9559` para NAOqi y no necesita Wi-Fi ni PC al ejecutar.

## PC → NAO por Ethernet

```powershell
python deploy/package.py --profile profiles/striker-NAO1.json --output dist/striker-NAO1.tar.gz
python deploy/deploy.py --archive dist/striker-NAO1.tar.gz --host 192.168.1.101
python deploy/deploy.py --archive dist/striker-NAO1.tar.gz --host 192.168.1.101 --apply
```

La segunda orden muestra las operaciones; `--apply` las ejecuta. Los archivos quedan en `/home/nao/PenaltyStack/releases/<hash>`, con un enlace `current`. No sobrescribe `/home/nao/Config`, `/home/nao/bin/bhuman` ni su servicio. El enlace anterior permite rollback: apuntar `current` al release previo. El contenido del paquete y el perfil tienen hashes SHA-256 en `manifest.json`; el comando de empaquetado imprime además el hash del tar.

## Arranque en el robot

Por SSH antes de la prueba: `cd /home/nao/PenaltyStack/current && sh deploy/run.sh`.
Para calibrar la visión con el balón Trionda real, añada
`--vision-log /home/nao/vision.jsonl`: cada 1 s se anexa una línea con rango,
rumbo y píxeles de la detección de `penalty/vision.py`. Copie el archivo al
terminar; el paquete no lo transmite ni lo borra. Dejar el proceso ejecutándose onboard; después desconectar Ethernet. El robot espera inmóvil hasta pulsar la parte frontal de la cabeza tras el silbato. Mantener pecho 0,3 s activa ABORT. El proceso termina al agotar 28 s o completar el rol. El lanzador se niega a iniciar si detecta el proceso `bhuman`: ALMotion/NAOqi y LoLA no deben competir por actuadores. Confirmar manualmente el perfil y la identidad de cada robot antes de ejecutar.

Actualmente `motion_enabled=false` deja todo movimiento físico apagado. Para ensayos controlados de marcha, ponerlo en true y establecer `PENALTY_ENABLE_WALK=I_UNDERSTAND` en la sesión onboard. La patada real no se ejecuta ni siquiera en ese modo. Si aparece `KICK_REQUEST_NOT_EXECUTED`, la prueba no es un penal funcional; no contarla como éxito.
