# Arquitectura y flujo

## Contratos

`Observation` contiene reloj, START, ABORT, distancia y rumbo al balón, edad de la observación y rumbo a la portería. `PenaltyController` produce `Command` con marcha, cabeza, petición de patada y lado de bloqueo. El controlador no importa MuJoCo ni NAOqi.

```text
MuJoCo: scene.xml + URDF V6 → simulador/observación → FSM → comando → adaptador MuJoCo
NAO: cámara+ALMemory → visión/observación → FSM → comando → ALMotion
RL futuro: JointSpaceEnv → objetivos articulares → dinámica MuJoCo
```

La cámara onboard procesa RGB localmente y detecta la mayor componente naranja compacta. Es un baseline inspirado en los candidatos naranja del repositorio de origen, no una migración binaria del detector Trionda; los umbrales y el cálculo de distancia aún requieren dataset y calibración del balón de 65 mm. El rumbo a portería del adaptador físico es el rumbo inicial configurado; solo vale si la colocación inicial se comprueba en pista. No hay percepción real de postes.

## Tirador

`WAIT_FOR_START → SEARCH_BALL → APPROACH_BALL → ALIGN_TO_BALL → ESTIMATE_GOAL → KICK → RECOVER → STOP`.

- El START exige pulsación frontal de cabeza durante 0,25 s. El reloj de 28 s comienza entonces.
- Un balón con edad superior a 0,25 s no autoriza acercamiento o patada.
- El tirador limita velocidad, rumbo y avance comandado a 0,6 m; solo pide una patada cuando el balón está a ≤0,22 m y el rumbo de balón/portería está dentro de tolerancias. Después recupera durante 1,5 s y se detiene. El avance comandado no reemplaza odometría real calibrada.
- ABORT en pecho durante 0,3 s se enclava y detiene marcha. El timeout lleva a `STOP`.

## Portero

`WAIT_FOR_START → TRACK_BALL → BLOCK → STOP`.

El MVP clasifica izquierda/centro/derecha por rumbo y se mantiene plantado; aún no realiza desplazamientos ni saltos. ABORT y límite de 28 s son comunes.

## Simulador

La escena contiene cancha de 2,4 × 2,4 m, marca a 1,3 m de la línea de gol, tirador 0,2 m detrás de la marca, arco de 1,5 m de ancho y 0,8 m de alto, área de portero de 0,6 × 2,2 m, líneas negras, postes azules, red blanca visual y balón naranja de 65 mm/55 g. `penalty/nao_model.py` carga dos instancias del URDF NAO H25 V6. Conserva las articulaciones principales, límites e inercias disponibles en el original, acopla los dos HipYawPitch y genera 23 servos por robot. Fija las manos y los dedos, que carecen de inercia en el URDF. Sus mallas visuales Webots `.mesh` se sustituyen por primitivas aproximadas; se añade geometría de pie para contacto y se evitan autocolisiones entre primitivas superpuestas del URDF. El archivo URDF original no se modifica.

`PenaltySimulator` fija la raíz de cada NAO a mocap y la mueve con comandos de velocidad ideales. MuJoCo calcula balón y articulaciones; la patada sigue siendo un impulso ideal. Comprueba secuencia, coordenadas y resultado aproximado, pero no valida marcha ni contacto de patada. `JointSpaceEnv` utiliza raíces libres y acciones articulares limitadas por URDF para futuros entrenamientos; con objetivos cero el NAO cae y no existe controlador de equilibrio, política ni recompensa. Ningún modo está calibrado contra un robot real.

## Origen seleccionado

`baseline/motion_manifest.json` registra 266 archivos del módulo motriz B-Human y su cierre de cabeceras, soporte sensorial y configuración, tomados del commit de origen. Sirven para preparar una comparación de trazas con una futura política RL; no se cargan durante la simulación ni por el proceso NAOqi. `heritage/manifest.json` conserva la selección inicial de keyframes y percepción. `heritage/upstream/License.md` conserva la licencia B-Human. El despliegue sigue aislado y no empaqueta `baseline/` ni `models/`.
