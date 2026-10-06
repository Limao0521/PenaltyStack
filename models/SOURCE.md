# Modelo NAO H25 V6

`NAOH25V60.urdf` es una copia sin modificaciones de
[`Wenao/protos/NAOH25V60.urdf`](https://github.com/RajasJoshi/UoG_Robotics_TDP4/blob/8a0949302aac5d610339fa02f3e4cfc437abb17d/Wenao/protos/NAOH25V60.urdf),
revisión `8a0949302aac5d610339fa02f3e4cfc437abb17d`.
SHA-256: `327dce62542ecd1327acf3fa7c5ec7b59274d44ce6c09a9ade3417cc39f132d6`.
La licencia del repositorio de procedencia es Apache 2.0; copia en `UPSTREAM_LICENSE`.

El robot del archivo se identifica como `NaoH25V60`: 131 links, 26 juntas
`revolute`, 16 `continuous` y 88 `fixed`. Se usa este archivo porque el URDF
`naoqi_driver2/share/urdf/nao.urdf` se identifica como `NaoH25V40`, otra
revisión de hardware.

`penalty/nao_model.py` transforma el URDF **en memoria** al cargar MuJoCo:

1. Elimina las referencias visuales a mallas Webots `.mesh`, no distribuidas
   con este archivo, y agrega primitivas de visualización aproximadas.
2. Fija manos y dedos sin inercia propia en el URDF. Conserva las 24 juntas
   restantes por robot y acopla LHipYawPitch/RHipYawPitch, dejando 23 acciones.
3. Agrega cajas de contacto a los pies. Desactiva contactos dentro de cada
   robot porque las primitivas de colisión originales se solapan en reposo.
   Se conservan contactos con la cancha, el balón y objetos de la cancha.
   El contacto entre los dos NAO también se omite en esta versión.

La topología, límites y masas que sí aparecen en el URDF vienen del archivo
original. Las simplificaciones anteriores impiden tomar este modelo como
validación dinámica del NAO físico. No hay mallas visuales originales,
calibración de fricción, motores ni control de equilibrio identificados en
hardware.
