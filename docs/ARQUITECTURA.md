# Arquitectura y flujo

## Contratos

`Observation` contiene reloj, START, ABORT, distancia y rumbo al balón, edad de la observación y rumbo a la portería. `PenaltyController` produce `Command` con marcha, cabeza, petición de patada y lado de bloqueo. El controlador no importa MuJoCo ni NAOqi.

```text
MuJoCo: scene.xml → simulador/observación → FSM → comando → adaptador MuJoCo
NAO: cámara+ALMemory → visión/observación → FSM → comando → ALMotion
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

La escena contiene cancha de 2,4 × 2,4 m, marca a 1,3 m de la línea de gol, tirador 0,2 m detrás de la marca, arco de 1,5 m de ancho y 0,8 m de alto, área de portero de 0,6 × 2,2 m, líneas negras, postes azules, red blanca visual y balón naranja de 65 mm/55 g. Hay dos modelos visuales aproximados de NAO. Los cuerpos del robot son cinemáticos y la pelota sigue la física de MuJoCo; la patada es un impulso idealizado. Por ello la simulación valida estados, coordenadas y trayectoria aproximada, pero no equilibrio, contacto pie-pelota ni límites de actuadores.

Las proporciones visuales principales (torso y cabeza) toman como referencia `Config/Scenes/Includes/NaoV6H25.rsi2` del commit de origen; la escena fue escrita de nuevo en MJCF y no incorpora SimRobot ni su modelo dinámico.

## Origen seleccionado

`heritage/manifest.json` registra hashes de los keyframes `kickForwardFast{,Long}.kmc`, `kickInfo.cfg` y las implementaciones de candidatos/percepción Trionda para adaptar y calibrar. Estos archivos no se cargan durante la simulación ni por el proceso NAOqi. `heritage/upstream/License.md` conserva la licencia del código fuente. El despliegue es nuevo para evitar el `rsync --del` del origen.
