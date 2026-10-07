# Simulación interactiva de penales

## Abrir en este equipo

```powershell
cd C:\Users\limao\Workspace\02_Research\PenaltyStack
.\.venv\Scripts\python.exe -m penalty.simulator
```

Abre el visor MuJoCo y el panel de control. La escena aparece inmóvil en
`PREPARED`; pulsa **Iniciar**. El runtime nativo ya está compilado en este
checkout. Cierra cualquiera de las dos ventanas para terminar; el trabajador
recoge los dos procesos B-Human. La escena permanece abierta al terminar un ensayo.

Para instalar desde una copia nueva: crear `.venv`, instalar
`requirements-sim.txt` y ejecutar `tools/build_motion_runtime.py` con Visual
Studio 2022 Build Tools, componente Desktop development with C++, y CMake.
La primera compilación descarga dependencias con SHA256 fijado. El build
posterior y la simulación utilizan los archivos locales.

## Controles

| Control | Funcionamiento | Atajo |
|---|---|---|
| Iniciar | Estabiliza ambos NAO y comienza el ensayo | Espacio |
| Pausa / Continuar | Congela/reanuda física, reloj, FSM y motor | Espacio |
| Reiniciar | Escena original y motores nuevos; queda pausado | R |
| Repetir | Reinicia y comienza un ensayo nuevo | Enter |
| Paso 12 ms | Un ciclo motriz y seis pasos físicos; permanece pausado | . |
| ABORT | Enclava STOP, solicita stand, espera hasta 2 s y congela el ensayo | A |
| Velocidad | 0.25, 0.5, 1 o 2 veces el ritmo de pared | Panel |
| Modo | Penal automático o acciones manuales; cambiarlo reinicia | Panel |
| Robot | Tirador o portero para acciones manuales | Panel |

Los atajos funcionan en el visor y el panel. Al escribir números en un campo
del panel se suspenden sus atajos. La velocidad conserva el paso físico de
2 ms y el ciclo motriz de 12 ms; si el PC no alcanza el ritmo solicitado,
el tiempo simulado avanza más despacio que el de pared.

En modo manual, selecciona robot y pulsa Pararse, Caminar, Patear, Bloquear o
Levantarse. Después pulsa Iniciar o reanuda. Avance/lateral usan m/s
(máximo ±0.15); giro rad/s (±0.35). Se puede elegir patada izquierda/derecha,
bloqueo centro/izquierda/derecha y ángulos de cabeza. Una petición de patada
se mantiene hasta su confirmación de finalización original; pulsar Patear
de nuevo solicita otro evento. **Los saltos laterales se recuperan
automáticamente**: termina el keyframe original, entra GetUp y el portero
vuelve a stand. Pulsar Bloquear de nuevo inicia otro salto. El bloqueo central
mantiene su guardia; Pararse o Levantarse solicita salir de ella. Levantarse
también permite solicitar recuperación manual.

El campo **Alcance pedido de patada (mm)** acepta 1500–3500; valor inicial
2500. Es una petición al conversor de potencia original, no una garantía de
distancia real. El caso inicial de 2500 mm está verificado para ambos pies;
otros valores y colocaciones son experimentos que deben medirse.

## Flujo y relojes

`PREPARED → SETTLING → RUNNING → FINISHED`. Pausa conserva el estado previo.
ABORT usa `ABORTING → ABORTED`; ERROR detiene el ensayo y permite reiniciar.

1. Cada ensayo crea dos procesos nativos con identidad, semilla y revisión.
2. Se pide stand. El asentamiento tarda al menos 2 s, como máximo 5 s;
   exige ambos pies cargados, inclinación de torso <10 grados y giro
   <0.2 rad/s durante 0.5 s. Si falla, aparece ERROR.
3. Comienza el reloj activo de 28 s. La FSM se actualiza cada 100 ms;
   el motor cada 12 ms y la física cada 2 ms.
4. El tirador espera el pulso de patada terminada antes de RECOVER.
   El portero solicita el primer bloqueo cuando el balón se acerca y lo
   conserva para ese disparo. El bloqueo central se mantiene hasta terminar
   el ensayo o ABORT; un lateral completa salto y recuperación una sola vez.
   El cambio de petición no trunca el salto: el motor original decide cuándo
   puede cambiar de fase. Cambios de rumbo o un balón parado no disparan
   otro salto para la misma pelota.
5. La escena se congela al terminar. Pausa, preparación, asentamiento y
   transición de ABORT no consumen el tiempo activo. Repetir inicia otro
   ensayo con ambos motores y relojes restablecidos.

## Ejecución sin ventana

```powershell
# Penal automático completo
.\.venv\Scripts\python.exe -m penalty.simulator --headless

# Marcha original, cinco segundos activos, balón apartado
.\.venv\Scripts\python.exe -m penalty.simulator --headless --mode manual --action walk --seconds 5 --forward 0.06 --ball-x 1 --ball-y 0.7

# Patada izquierda original y contacto físico
.\.venv\Scripts\python.exe -m penalty.simulator --headless --mode manual --action kick --seconds 6

# Patada derecha, alcance solicitado de 2500 mm
.\.venv\Scripts\python.exe -m penalty.simulator --headless --mode manual --action kick --kick-type forwardFastRight --kick-length-mm 2500 --seconds 6

# Bloqueo central del portero
.\.venv\Scripts\python.exe -m penalty.simulator --headless --mode manual --role keeper --action block --block-side center --seconds 4

# Salto lateral y recuperación automática; repetir con --block-side right
.\.venv\Scripts\python.exe -m penalty.simulator --headless --mode manual --role keeper --action block --block-side left --seconds 18

# Verificación completa; incluye abrir/cerrar visor en Windows
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`--seconds` acorta la ejecución headless; con ventanas se usa el límite de
28 s. `--ball-x`, `--ball-y`, `--seed`, `--kick-type`, `--kick-length-mm`, `--block-side` y
`--record-dir` fijan el ensayo. Las coordenadas de balón son metros en mundo.

## Motor y física utilizados

Los 559 archivos originales quedan en `baseline/bhuman`, con hashes. El
runtime ejecuta MotionEngine, WalkingEngine, KickEngine, KeyframeMotionEngine,
FallEngine y sus generadores/proveedores originales; no reproduce una traza
ni genera una marcha alternativa. [Detalle del host y protocolo](../native/bhuman_motion/README.md).

Cada NAO tiene raíz libre, 24 bisagras y 23 servos. Las manos no tienen
actuadores; los HipYawPitch comparten motor y medida. El sensor de IMU usa
el offset original y recibe gyro y aceleración específica locales. La
orientación viene del torso físico. FSR divide las fuerzas de contacto
pie–suelo entre cuatro posiciones originales por pie y convierte N a kg.
Las patadas mueven el balón por contacto; no se escriben sus velocidades.

El perfil actual `mujoco-truth-v2` corrige la orientación al vector angle-axis
que espera B-Human, sincroniza radio real de pelota (32.5 mm) y calibra la
fase de golpe derecho a 200 ms para reducir retraso del servo. Conserva los
endpoints originales, el IK, el balanceador y el golpe izquierdo de 100 ms.
Los bytes originales y la configuración KMC se mantienen intactos. Ver
[calibración y procedencia](../native/bhuman_motion/README.md).

Se usa `implicitfast`, 100 iteraciones del solver, masas, inercias y
ganancias/límites de esfuerzo de la conversión RSI3. Autocolisiones y
contactos entre robots siguen desactivados. La percepción de balón/portería
y orientación son ideales; las manos, corrientes, temperaturas, offset del
gyro y desgaste/predicción articular tienen entradas ideales/sintéticas
declaradas. No se simula la cámara, latencia real ni deformación del césped.

## Registros y alcance

Cada ensayo escribe `runs/<trial>.jsonl`: METADATA con hashes/revisión/semilla,
FRAME con sensores de entrada, petición, salida motriz, esfuerzos y estado
físico, y SUMMARY con resultado. Los reinicios crean archivos nuevos. El
resultado geométrico distingue gol, fuera e interacción de bloqueo;
los contadores de contacto y caída permiten comparar acciones aunque el
penal no termine en gol. `block_contacts` cuenta muestras de contacto,
no bloqueos independientes. `fallen` se enclava durante todo el ensayo.
`robots.<rol>.fall_state` informa la postura actual: el portero recuperado
puede estar `upright` con `fallen=true`, porque sí se tumbó durante el salto.
El panel muestra ambos datos para no presentar una caída histórica como
una recuperación fallida.

Esta integración proporciona un baseline ejecutable para comparar RL.
Ambas patadas tienen contacto físico durante la fase kick, una ejecución y
confirmación de finalización sin caída. Ambos saltos laterales alcanzan
GetUp, postura erguida, altura de torso >0.28 m y apoyo de ambos pies. La
recuperación supera el mismo criterio de estabilidad inicial durante 0.5 s
y permanece erguida un segundo adicional. No se ha implementado una política
RL; esta es la base calibrada para comparar el sistema que diseñes.

El binario Windows no se despliega al NAO. El paquete NAOqi mantiene su
patada deshabilitada: [requisitos para hardware](PREPARACION_NAO.md).

### Evidencia del 7 de octubre de 2026

Suite completa: **45 pruebas, 0 fallos**, 165.306 s.
La [validación de las correcciones](validation/simulation-fixes-2026-10-07.json)
incluye pruebas físicas, integración con visor/panel y repetición completa.
El [informe anterior](validation/simulation-2026-10-07.json) conserva los
fallos de contacto y recuperación del perfil anterior como evidencia histórica.

| Ensayo | Resultado observado |
|---|---|
| Stand 10 s y marcha 5 s | Ambos NAO sin caída; avance neto aproximado de 0.47 m |
| Patada izquierda | Una fase, confirmación, contacto durante kick y desplazamiento 1.014 m |
| Patada derecha | Una fase, confirmación, contacto durante kick y desplazamiento 0.962 m |
| Bloqueos centro/izquierda/derecha | Tres keyframes originales; laterales con recuperación automática |
| Recuperación lateral | Izquierda y derecha: GetUp → walk → stand, erguido con ambos pies apoyados |
| Penal completo, tres veces | 28 s activos, una patada y resultado blocked; sin caída física registrada |
| Repetición con semilla 0 | Diferencia máxima de qpos final: 0; procesos recogidos |
| Visor + panel Windows | Iniciar, pausa, paso de 12 ms, reiniciar, repetir y cerrar verificados |

Las pruebas exigen contacto pie–balón durante kick, no sólo finalización de
fase. Los resultados corresponden a este modelo, perfil, colocación inicial
y semilla 0 en este equipo; no certifican todo el rango de alcances ni hardware.

## Revisión de las fronteras de proceso

El panel envía comandos a una cola local; sólo el trabajador muta la física.
El ejecutable se inicia por ruta fija y lista de argumentos, sin shell ni
puertos de red. Los frames JSONL tienen límite de 64 KiB y se comprueban
revisión, identidad, secuencia, reloj, tamaños y valores finitos. Una salida
inválida detiene el ensayo antes de aplicarla. Timeout, cola llena y cierre
del visor deben recoger los hijos. Las descargas de compilación usan hashes
fijados y el ZIP se verifica contra escapes de ruta. No se incorporan
credenciales ni un controlador de juego.
