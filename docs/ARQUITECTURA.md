# Arquitectura y flujo

## Contratos

`Observation` contiene reloj, START, ABORT, distancia y rumbo al balón, edad de la observación y rumbo a la portería. `PenaltyController` produce `Command` con marcha, cabeza, petición de patada y lado de bloqueo. El controlador no importa MuJoCo ni NAOqi.

```text
MuJoCo: física/sensores → FSM → JSONL → B-Human C++ original → JointRequest → servos → física
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

La FSM clasifica izquierda/centro/derecha por rumbo. El adaptador MuJoCo solicita genuflexión central o salto lateral original cuando el balón se acerca. El adaptador físico NAOqi aún mantiene postura estática. ABORT y límite de 28 s son comunes.

## Simulador

La escena contiene cancha de 2,4 × 2,4 m, marca a 1,3 m de la línea de gol, tirador 0,2 m detrás de la marca, arco de 1,5 m de ancho y 0,8 m de alto, área de portero de 0,6 × 2,2 m, líneas negras, postes azules, red blanca visual y balón naranja de 65 mm/55 g. `tools/build_bhuman_model.py` convierte la descripción `NaoV6H25.rsi3` de B-Human a MJCF y 50 mallas OBJ usadas por el robot. Se trasladan los 25 cuerpos, 24 articulaciones con ejes y límites, masas/inercia, geometrías de colisión y parámetros de los servos que figuran en la fuente. `penalty/nao_model.py` carga dos instancias, acopla los HipYawPitch y expone 23 servos por robot. Las mallas son sólo visuales; para contacto se usan las primitivas del modelo B-Human. Las autocolisiones y los contactos entre robots se desactivan para este MVP.

`PhysicalSimulator` usa raíces libres y 23 servos por NAO, con límites de esfuerzo y ganancias originales del modelo. `SensorAdapter` lee ángulos, gyro, acelerómetro y fuerzas de contacto; deriva ocho cargas FSR en kg equivalente. La pose usada para percepción ideal se centra entre ambos pies como `SimulatedRobot3D::getRobotPose`; se distingue del origen del torso RSI3. Las manos no tienen actuadores y el HipYawPitch derecho comparte la medida del izquierdo.

`BHumanBridge` inicia dos procesos C++ con pipes locales acotados y JSONL validado. Cada proceso conserva Blackboard, módulos y fases originales; MuJoCo avanza seis pasos de 2 ms por ciclo de 12 ms. La FSM solicita acciones cada 100 ms de tiempo simulado y espera el pulso real de finalización de patada. No se inyecta trayectoria de raíz ni impulso al balón.

`SimulationSession` posee física, motores, relojes y registros. Tk y atajos envían comandos por cola; el panel lee copias de telemetría. El mutex del visor protege mutaciones físicas y nunca las esperas al proceso. Cada reinicio crea dos motores nuevos, semilla y ensayo nuevo. Ver [operación](SIMULACION.md). `JointSpaceEnv` conserva la interfaz de objetivos para RL, sin política ni recompensa. No existe calibración contra hardware.

## Origen seleccionado

`baseline/motion_manifest.json` registra 559 archivos originales: módulo motriz, cierre de cabeceras y unidades de implementación, soporte sensorial y configuración. El build verifica sus SHA256 antes de compilar. `heritage/manifest.json` conserva la selección inicial y `heritage/upstream/License.md` la licencia. El despliegue NAOqi permanece separado y no empaqueta el runtime C++ de Windows, `baseline/` ni `models/`.
