# Percepción B-Human del rival

**Actualización:** la ruta nueva de simulación ya estima orientación corporal
desde RGB y reconstruye cámara/odometría con encoders, IMU y FSR propios.
Ver [ORIENTACION_PORTERO.md](ORIENTACION_PORTERO.md). El detector original
B-Human sigue entregando cajas; esta extensión visual es propia. El informe
v1 de abajo y el observador NAOqi conservan sus resultados y yaw desconocido.

2026-10-10. **Detector activado y conectado a las observaciones del portero.**
Se ejecuta en un proceso local de imágenes, con la misma revisión original
`2360ab1100ad28f70e190c893f6b095651ff0bbc` que el motor de movimientos.
Hay dos entradas: red neuronal para la cámara superior y detector de
contraste para la inferior. El balón conserva su cámara inferior, su
seguimiento y la política `vision_keeper_v2` publicados.

La integración obtiene una **posición aproximada del robot próximo al
balón**, edad, covarianza e identificador de asociación local. La orientación
corporal sigue con `yaw_valid=false`. La política v2 sigue decidiendo por
la trayectoria del balón: esta entrega no agrega una mejora de atajadas
atribuible a postura, ni habilita movimientos en el NAO real.

## Componentes que se ejecutan

| Componente | Recorrido activo |
| --- | --- |
| Cámara superior | Modelo ONNX original `4_anchor_boxes_model_no_activation_20230629-220730.onnx`, ONNX Runtime 1.10.0 CPU, entrada YUV 80×60 |
| Preprocesamiento superior | `Resize::shrinkY`, `fillGrayscaleThumbnail`, `fillUVThumbnails` y `applyColorNetwork`, cuerpos originales compilados |
| Cajas superiores | `boundingBoxes`, `predictionToBoundingBox`, NMS 0.30 y supresión de cajas contenedoras originales; umbral 0.60 |
| Cámara inferior | `scanImage`, `classifyRegions`, `discardHomogeneousAreas`, `dbScan`, `expandCluster` y `regionQuery`, cuerpos originales compilados |
| Proyección | Extremo inferior de la caja → plano del suelo, con intrínsecos y transformación de la cámara propia; metros y covarianza |
| Asociación | Robot próximo al balón, compensación de odometría y continuidad; dos capturas independientes y coherentes para contraste inferior |
| Consumidor | `vision_state_v1` y `rl_obs_v2`, mediante las máscaras ya publicadas |
| NAOqi | Dos cámaras locales, sensores, START/ABORT y 28 s; registro de decisiones con `executed=false` |

La red superior de B-Human detecta cajas de NAO; sus salidas no incluyen
yaw ni clasificación de camiseta. Véase la [documentación original](https://docs.b-human.de/coderelease2023/perception/robot-detection/).
En el componente aislado también se ejecuta la rama inferior de la revisión
fijada, que detecta contraste y regiones mixtas en las piernas visibles.

`native/bhuman_vision/main.cpp` aporta el host y declaraciones de plataforma.
`tools/build_robot_perception.py` extrae los cuerpos de las funciones sin
modificarlos y genera los archivos bajo
`native/bhuman_motion/build/generated/vision/`. Los 19 originales, sus pesos,
configuración y licencias están en `heritage/bhuman_robot_perception/`, con
SHA-256 en `manifest.json`. Los 559 originales de movimiento y RSI3 conservan
sus bytes. DLL/.so se descargan al compilar desde URLs con revisión y SHA-256
fijados; no se descargan durante el penal.

## Adaptaciones y medidas desconocidas

El componente incluye estas adaptaciones explícitas:

- RGB se codifica a YUYV con la conversión BT.601 original; también admite
  directamente YUYV. El recorrido NAOqi actual captura RGB.
- La saturación independiente de iluminación de la imagen inferior tiene
  un adaptador escalar de la fórmula de `ECImageProvider`; puede diferir de
  las aproximaciones recíprocas del JIT original.
- No se ejecutan los proveedores de borde de campo, contorno propio ni
  camiseta. La rama inferior usa `trimObstacles=false`; las cajas pasan
  después por los controles de visibilidad, anchura y rango de la proyección.
- La selección supone el escenario de penal con un rival próximo al balón.
  El tipo permanece `unknown`; no se afirma reconocimiento de equipo.
- Las regiones inferiores no tienen probabilidad neuronal. El `1.0` de
  candidato en el mensaje nativo está acompañado de `confidence_measured=false`;
  la observación del rival exporta `confidence=null`. Requieren dos capturas
  separadas por hasta 250 ms y con desplazamiento de hasta 25 cm.
- La posición corresponde aproximadamente al borde visible de contacto con
  el suelo. Puede diferir del centro de los pies/cuerpo; no es una pose de
  esqueleto. Caída y orientación corporal no se miden en esta integración.

Se rechazan pies fuera de cuadro, rayos sobre el horizonte, cámaras inválidas,
anchuras/rangos incompatibles, cuadros repetidos o fuera de orden, edad mayor
de 250 ms y sensores no sincronizados. Una pérdida limpia las máscaras;
la posición previa no se presenta como una nueva detección. La velocidad
del rival no se transforma en orientación.

El intercambio `bhuman_robot_pixels_v1` usa tuberías locales acotadas,
secuencia, timestamp de captura y procedencia. El proceso no tiene interfaz
de actuadores, LoLA, SimRobot, GameController ni sockets de juego. Timeout,
canal roto y cola llena también terminan y recolectan el hijo.

## Verificación desde píxeles

El ensayo reproducible `bhuman_opponent_pixel_probe_v1`, semilla 66002,
capturó 30 imágenes de ambas cámaras de cabeza cada 120 ms, sobre la cancha
existente y con movimientos/raíces/contactos físicos del simulador:

| Medida | Resultado |
| --- | ---: |
| Cajas de contraste inferior | 30/30 capturas |
| Posición del rival validada tras confirmación | 28/30 |
| Orientación del rival validada | 0/30 |
| Error mediano respecto al centro de los pies | 11.4 cm |
| Percentil 95 del error | 12.4 cm |
| Inferencia superior, mediana en este PC | 5.13 ms |
| Procesamiento inferior, mediana | 1.33 ms |

Son **medidas preliminares de integración** con un rival parado/preparándose.
Extrínsecos, odometría e IMU propios son ideales del simulador, los
intrínsecos nominales y el tiempo de cómputo no retrasa el reloj simulado.
No representan precisión o latencia del NAO real.

La red superior no aceptó cajas en esas 30 imágenes de MuJoCo. En una imagen
publicada por B-Human aceptó un NAO con confianza 0.710 frente a umbral 0.600.
El reporte conserva URL, hash, recorte y escala de esa comprobación;
no calcula distancia de una foto sin calibración.

También se ejecutaron 16 episodios de integración, uno por cada uno de los
ocho estilos y portero, semilla 66001, sin errores de simulación. Se validó
posición en 1080/1142 cuadros; el error mediano respecto al centro de los pies
fue 22.8 cm durante los episodios. Las piernas en movimiento y el borde de
contacto requieren calibración/validación adicional. Esa muestra comprueba
el recorrido de imágenes y motores; no demuestra una ganancia competitiva. El resultado
anterior de 74/80 sigue perteneciendo al ensayo de balón por cámara.

Reportes, trazas con etiquetas de evaluación separadas y imágenes intactas:
[bhuman_opponent_20261010](validation/vision/bhuman_opponent_20261010/).
La observación consulta únicamente imágenes y sensores propios. Las
coordenadas reales del balón/rival se leen fuera del recorrido de decisión
para medir errores; una prueba bloquea esas lecturas dentro de la percepción.

## Ejecutar en PC

```powershell
# Compilar el proceso de percepción; detecta MSVC 2022 o 2026
.\.venv\Scripts\python.exe tools/build_robot_perception.py

# Imagen local VGA/QVGA sin espejo; no infiere rango sin calibración
.\.venv\Scripts\python.exe tools/detect_bhuman_robots.py --image imagen.png --output runs/detecciones.json

# Cámaras de cabeza, proyección y asociación desde píxeles
.\.venv\Scripts\python.exe tools/validate_robot_perception.py --frames 30 --seed 66002 --output runs/vision_bhuman

# Ensayo nuevo con ambas cámaras y los dos detectores activos
.\.venv\Scripts\python.exe tools/evaluate_keeper_vision.py --opponent-detector bhuman --episodes 1 --workers 4 --seed 66001 --output runs/vision_bhuman/duelos.json

.\.venv\Scripts\python.exe -m unittest tests.test_robot_perception -v
```

`--opponent-detector none` conserva el experimento publicado de balón por
cámara. Los contratos, políticas, recompensas y reportes anteriores
permanecen versionados sin cambiar sus definiciones.

## NAO real

`profiles/keeper.bhuman-vision.example.json` selecciona el detector y
`penalty/nao.py` despacha a `nao_opponent_vision.run`. La cámara inferior
QVGA alimenta balón y contraste; la superior VGA alimenta la red. Cada
cámara conserva su timestamp, transformación y edad de sensores. Las lecturas
a recepción son aproximadas y se invalidan si están a más de 80 ms de la
captura. La edad incluye también el tiempo de inferencia.

El modo NAOqi solo registra. La cabeza debe colocarse para que las cámaras
puedan observar la escena; este modo no mueve cabeza, marcha ni brazos.
Faltan medir intrínsecos/extrínsecos, sincronización, distorsión, cobertura de
imagen y latencia del robot concreto. Los ceros/`TO_FILL` del perfil deben
reemplazarse por esas medidas.

En este PC se compiló y ejecutó Windows x64. Para el NAO V6 hace falta compilar
el host con una toolchain Linux x86-64 compatible con su sistema y comprobar
su HELLO/modelo/latencia onboard. El CMake admite Linux; no se ha compilado ni
ejecutado ese destino en esta entrega. Su biblioteca ONNX Runtime fijada está
identificada en el manifiesto.

El empaquetado requiere `--vision-runtime /ruta/bhuman_vision` y la biblioteca
`libonnxruntime.so.1.10.0` fijada a su lado. Rechaza un `.exe` Windows y perfiles
con movimientos/bloqueos habilitados. Incluye módulos Python 2.7, host, pesos y
licencias, sin NumPy/Torch/MuJoCo. El lanzador mantiene el control existente
de exclusión con B-Human y la ruta de despliegue por Ethernet.

This product includes software developed by B-Human (http://www.b-human.de).
Licencia original: [License.md](../heritage/bhuman_robot_perception/License.md).
