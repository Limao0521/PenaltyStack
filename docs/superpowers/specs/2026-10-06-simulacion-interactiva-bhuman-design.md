# Simulación interactiva y motor B-Human en MuJoCo

Fecha: 2026-10-06. Estado: diseño conversacional aprobado; especificación pendiente de revisión del usuario.

## Objetivo y alcance

El usuario quiere controlar los ensayos de penales y ejecutar el motor de movimiento B-Human dentro de la simulación. El resultado debe permitir iniciar, pausar, repetir e inspeccionar un ensayo, con marcha, patada y bloqueo generados por el motor original y ejecutados por la dinámica de MuJoCo. También debe conservar una referencia reproducible para comparar posteriormente con RL.

Esta entrega comprende el panel de operación, el runtime motriz C++, el puente de sensores/actuadores, la conexión con los dos roles y la documentación de compilación y uso. La preparación del motor para el NAO real requiere otra integración y validación física; el éxito de esta entrega se mide en simulación.

## Evidencia del estado actual

- `penalty/simulator.py` avanza mediante raíces mocap y aplica directamente velocidad al balón al recibir una petición de patada. Su controlador se actualiza cada 100 ms.
- `penalty/joint_env.py` expone objetivos articulares con raíces libres, pero no ejecuta un controlador de equilibrio.
- `baseline/motion_manifest.json` conserva 266 archivos del origen `SabanaHerons2026`, revisión `2360ab1100ad28f70e190c893f6b095651ff0bbc`. Los algoritmos de movimiento todavía no se compilan ni ejecutan aquí.
- `tools/import_motion_engine.py` busca includes bajo `Src/` y junto al archivo que los incluye. El origen también usa `Src/Libs/`; por tanto, la selección actual no cierra las dependencias del framework ni las unidades de implementación asociadas.
- En el origen, `Src/Libs/MathBase/Constants.h` define un ciclo motriz de 0,012 s. `MotionEngine` consume `MotionRequest`, coordina generadores/fases y produce `JointRequest`.
- El modelo visual/mecánico NAO V6 procede de B-Human 2025, revisión `d89d603f0388208cecd1c3144b6e8eb92aef4a93`. Esta procedencia es distinta de la del motor migrado y se registrará por separado.
- El PC tiene Visual Studio 2022 Build Tools con herramientas C++ y CMake. La implementación comprobará sus versiones y requisitos al configurar la compilación.

## Arquitectura elegida

MuJoCo será el único motor de física. Python gestionará escena, observaciones de penales, sesión, panel y registros. Un ejecutable C++ nativo ejecutará el subsistema motriz B-Human; habrá un proceso independiente para cada NAO.

```text
Panel de operación → sesión de ensayo → FSM de tirador/portero
                                  ↓ petición motriz
MuJoCo → sensores → puente local → B-Human C++ → ángulos y rigidez → MuJoCo
```

El proceso independiente evita compartir blackboards, generadores y variables globales entre los dos robots. La comunicación será por stdin/stdout, sin servidor de red. Los mensajes de diagnóstico del ejecutable se enviarán por stderr. La opción de una DLL fue considerada; el aislamiento por proceso simplifica el reinicio completo de cada ensayo y el diagnóstico de fallos.

La escena y el panel se abrirán mediante un único comando de Python. El panel utilizará Tkinter y la vista 3D utilizará el visor MuJoCo existente. Los comandos del panel pasarán por una cola al hilo propietario de la simulación; las callbacks de interfaz no modificarán directamente el estado físico. Las lecturas y actualizaciones del visor respetarán su lock.

## Panel y estados de sesión

La escena se abre pausada y muestra los controles aunque el runtime no esté disponible. En ese caso informa el error y deshabilita la ejecución B-Human hasta corregirlo.

| Control | Comportamiento |
| --- | --- |
| Iniciar | Inicia un ensayo nuevo desde la escena preparada. Realiza el asentamiento de postura y después activa START y el reloj del penal. |
| Pausar / Continuar | Congela o reanuda física, motor, FSM y reloj del penal conservando el estado. |
| Reiniciar | Restablece escenario, controladores y procesos B-Human; queda preparado y pausado. |
| Repetir | Restablece completamente el ensayo y vuelve a iniciarlo con los mismos parámetros y semilla. |
| Avanzar un paso | Desde pausa, ejecuta un ciclo motriz de 12 ms y sus pasos físicos; queda pausado. |
| Abortar | Enclava ABORT en ambos roles, detiene las peticiones activas y termina el ensayo. Reanudar no elimina el enclavamiento. |
| Velocidad | Selecciona 0,25×, 0,5×, 1× o 2×; modifica la espera de reproducción, conservando el intervalo físico y motriz. |
| Rol y acción manual | Selecciona tirador/portero y permite postura, marcha, patada y bloqueo para ensayos aislados. |

Atajos: Espacio para iniciar o pausar/continuar según el estado; R para reiniciar; Enter para repetir; punto para avanzar un paso; A para abortar. Los mismos comandos deben funcionar desde el panel y desde la vista 3D cuando tenga el foco.

Estados explícitos: PREPARED, SETTLING, RUNNING, PAUSED, FINISHED, ABORTED y ERROR. PAUSED conserva su estado anterior. FINISHED y ABORTED conservan la escena visible. ERROR congela el ensayo y muestra una causa recuperable mediante reinicio cuando sea posible. Cerrar cualquiera de las ventanas finaliza la sesión, cierra el visor y recoge los dos procesos hijos.

El panel muestra rol, estado de la FSM, petición motriz, fase ejecutada, tiempo físico, tiempo activo del penal, velocidad de reproducción, apoyo de pies, caída y resultado. Las acciones manuales registran el modo manual y no se mezclan con las decisiones automáticas de la FSM. Al cambiar entre modo manual y penal se inicia una sesión preparada nueva.

La configuración inicial permite elegir pie de patada, velocidad de marcha manual y lado de bloqueo. Los ensayos de esta entrega usan la cancha y el balón ya documentados; se conservan esos valores como configuración predeterminada.

## Migración y compilación del motor original

Se conserva la revisión `2360ab1100ad28f70e190c893f6b095651ff0bbc` como motor de referencia. La migración incorpora las unidades `.cpp` y `.h` del framework, streaming, matemáticas, plataforma y herramientas que realmente necesita el subsistema motriz, junto con configuración y dependencias externas fijadas. La resolución de includes debe contemplar `Src/`, `Src/Libs/` y rutas relativas, y el cierre de includes debe complementarse con el cierre de enlazado.

Los archivos originales se preservan con hashes y licencia. El adaptador, la selección de proveedores y el target de compilación viven fuera de los algoritmos originales. Cualquier modificación imprescindible en código de origen se mantiene como parche explícito y se documenta; no se sustituye la marcha por una implementación Python.

El runtime incorpora `MotionEngine`, `WalkingEngine`, `WalkAtSpeedEngine`, `WalkToBallAndKickEngine`, `WalkKickEngine`, `KickEngine`, `HeadMotionEngine`, `ArmKeyFrameEngine`, `KeyframeMotionEngine`, `FallEngine` y los proveedores sensoriales necesarios para sus dependencias. Los demás archivos motrices conservados siguen disponibles como referencia; sólo los proveedores seleccionados para este escenario se ejecutan.

Se usará el mecanismo de módulos/representaciones de B-Human y una configuración de proveedores dedicada al ensayo. Las entradas que en el sistema completo vendrían de percepción o hardware se suministran mediante el adaptador de simulación. Las representaciones predeterminadas sólo se permiten para funciones ajenas al ensayo, con su lista explícita. Apoyo, orientación, posición de articulaciones y tiempo se actualizan desde la simulación.

Un script configura y compila el ejecutable con CMake y Visual Studio Build Tools. Detecta requisitos ausentes y comunica cómo resolverlos. Después de compilar, iniciar la simulación no depende del checkout externo usado para migrar el código. El target motriz no enlaza SimRobot ni incorpora GameController, interfaz de equipo o runtime de percepción completo.

## Contrato del puente local

El protocolo tendrá versión, límites de tamaño y mensajes JSON delimitados por línea. Sólo acepta mensajes completos con tipos, longitudes y valores finitos validados. Cada mensaje identifica ensayo, rol y secuencia para impedir reutilizar una salida de otra sesión.

| Mensaje | Contenido |
| --- | --- |
| HELLO | Versión del protocolo, revisión/hashes del motor, lista y orden de articulaciones, intervalo motriz y movimientos disponibles. |
| INIT | Identidad del robot/rol, semilla y perfil de parámetros del ensayo. |
| STEP | Secuencia, tiempo motriz, sensores, posición relativa del balón y peticiones de cuerpo/cabeza/brazos. |
| OUTPUT | Secuencia correspondiente, ángulos solicitados, rigidez, odometría, fase ejecutada, movimiento activo y finalización. |
| ERROR | Código y descripción del fallo; no se aplica como comando articular. |
| SHUTDOWN | Terminación ordenada del runtime. |

INIT crea la configuración motriz antes de aceptar STEP. Cada STEP tiene exactamente una OUTPUT válida o un fallo explícito. El consumidor establece timeout de 2 s por respuesta; un timeout detiene el ensayo en ERROR, sin extrapolar órdenes. Un ciclo lento afecta la velocidad de reproducción real, no el intervalo simulado. La UI permanece receptiva durante el intercambio con los procesos.

La creación de procesos usa rutas verificadas y argumentos estructurados, sin shell. Se valida el HELLO antes de actuar. Las secuencias duplicadas, tamaños incorrectos, salida no finita y cierre inesperado del proceso llevan a ERROR. El reinicio recrea ambos procesos y descarta colas/respuestas de la sesión previa.

## Sensores, articulaciones y unidades

- MuJoCo usa metros y segundos; las magnitudes B-Human que los requieren se convierten a milímetros y milisegundos. Ángulos y velocidades angulares conservan radianes y radianes/s. La aceleración y fuerza se convierten según el contrato concreto de las representaciones de origen.
- La tabla de mapeo es explícita por nombre y fija el signo/eje de cada articulación. B-Human representa manos y ambos HipYawPitch; el modelo actual tiene 23 actuadores por robot. Las manos sin geometría actuada se marcan como no modeladas y el HipYawPitch derecho comparte la salida izquierda mediante el acoplamiento existente.
- `SensorData::off`, valores de ignorar y rigidez predeterminada se interpretan antes de aplicar objetivos. La rigidez cero desactiva el esfuerzo del servo; no deja activo un controlador de posición residual.
- Se proporcionan ángulos medidos, orientación de torso, velocidad angular y aceleración específica en el marco esperado por B-Human. Los sensores IMU se sitúan en el torso; los ejes y la gravedad se verifican con posturas conocidas.
- Las ocho lecturas FSR se obtienen de las fuerzas de contacto de cada pie y su distribución respecto a las posiciones FSR. Se documentan unidades, reparto y aproximación; no se inventan cambios de pie de apoyo por un temporizador.
- Las temperaturas, corrientes o desgaste que MuJoCo no reproduce se declaran como entradas sintéticas en el perfil. Esas entradas no se usan como evidencia de fidelidad al hardware.
- Las observaciones de balón/portería para la FSM y las peticiones motrices usan geometría de la escena. Esta entrega prueba control con observación ideal; no valida percepción por cámara.

## Dinámica y sincronización

El modo B-Human utiliza raíces libres: `nmocap == 0`. Las raíces se posicionan únicamente al preparar el escenario. Durante la ejecución no se sobrescribe su trayectoria y no se aplica un impulso ideal al balón. Los servos actúan con límites angulares, de velocidad y esfuerzo registrados en el perfil del modelo.

La física mantiene pasos de 2 ms; cada ciclo B-Human de 12 ms comprende seis pasos físicos. La FSM puede conservar su periodo de 100 ms mediante un programador basado en tiempo simulado; las peticiones quedan retenidas entre actualizaciones. La publicación visual se desacopla del control, con objetivo de 30 imágenes/s cuando el PC lo permita.

El asentamiento inicial dura al menos 2 s y hasta 5 s de tiempo físico, fuera del presupuesto del penal. Se acepta la postura cuando ambos pies soportan al robot, el torso está dentro de 10° de la vertical y sus velocidades angulares están por debajo de 0,2 rad/s durante 0,5 s. Si no se estabiliza, se informa el fallo; no comienza automáticamente el penal con un robot caído.

Pausar no consume tiempo físico, motriz ni del penal. El periodo activo del penal se mantiene en 28 s. Avanzar un paso después de START sí consume 12 ms de ese periodo. Cambiar la velocidad de reproducción sólo cambia las esperas contra el reloj de pared.

El contacto pie-balón mueve la pelota y el cuerpo/las extremidades del portero la interceptan. Las diferencias de masa, fricción, ganancias y contactos entre el modelo y el hardware se registran. Cualquier ajuste de estabilidad vive en un perfil versionado y se acompaña de su motivo.

## Tirador, portero y finalización de acciones

La FSM del tirador solicita marcha y alineación; el puente convierte esas peticiones a `MotionRequest`. Al entrar en KICK se enclava una única solicitud y se selecciona el tipo/pie disponible en la configuración original. La fase motriz se mantiene hasta confirmar ejecución o fallo. RECOVER y STOP deben respetar la finalización física; se revisará el salto inmediato KICK → RECOVER actual para evitar cancelar una patada en curso.

El portero usa postura preparada y las variantes de bloqueo izquierda, centro y derecha que realmente estén disponibles en el generador de keyframes del origen. Se evitará reiniciar una misma acción cada tick: la nueva petición requiere finalizar la anterior o una transición permitida por el motor. La etiqueta observada y la acción ejecutada se muestran por separado.

Si el origen no contiene una variante solicitada, el panel la muestra como no disponible y la integración no se declara completa para ese criterio. El diagnóstico identifica la dependencia o configuración ausente y se resuelve antes de cerrar la entrega. La detección de caída y recuperación automática usa las fases originales cuando estén configuradas.

ABORT detiene las nuevas peticiones de marcha/patada y solicita transición motriz a una postura permitida antes de congelar la escena. Se permite hasta 2 s simulados para finalizar esa transición, fuera del reloj activo del penal. Si no termina, la escena se congela con la fase observada y el registro identifica esa condición. Pausa es una congelación inmediata; no se interpreta como un ABORT.

## Registro y comparación con RL

Cada ensayo guarda un registro con revisión del motor y del modelo, hashes de configuración, semilla, modo, unidades y perfil de entradas sintéticas. Las muestras motrices contienen secuencia/tiempo, sensores, peticiones, ángulos/rigidez, pose/velocidad de la raíz, contactos, movimiento ejecutado y estado de la FSM.

El resumen registra gol/bloqueo/fuera/sin resultado, caídas, tiempo activo, distancia recorrida y número de acciones ejecutadas. El resultado de gol se enclava al cruzar el plano de la portería dentro de la abertura, para que el movimiento posterior del balón no borre el evento. Las métricas mecánicas incluyen esfuerzo de actuadores; la energía eléctrica de hardware no se infiere de esa medida.

Repetir conserva semilla, configuración y condiciones iniciales, pero crea nueva identidad de ensayo. Un futuro controlador RL podrá usar el mismo puente de sensores/actuadores y escenario; entrenar esa política queda para la etapa posterior.

## Comprobación y criterios de aceptación

1. Compilar el ejecutable desde las fuentes migradas y recibir HELLO con la revisión esperada. Comprobar que los proveedores originales calculan las salidas; copiar archivos o reproducir una traza no satisface este criterio.
2. En GUI, abrir pausado; iniciar, pausar durante 2 s de pared, avanzar un ciclo, continuar, reiniciar, repetir y abortar. Comprobar tiempo, receptividad del panel y recogida de procesos al cerrar cualquiera de las ventanas.
3. Verificar mapeo por nombres, unidades y signos, HipYawPitch compartido, valores especiales y rigidez cero. Inyectar salida inválida/timeout para comprobar ERROR sin aplicar comandos corruptos.
4. Con raíces libres, sostener la postura 10 s sin caída ni apoyo artificial. Una caída se identifica con inclinación del torso superior a 45° durante 0,2 s o contacto torso/cabeza-suelo, registrando qué condición se activó.
5. Ejecutar marcha conservadora hacia delante y lateral durante 5 s por dirección. Debe existir movimiento articular y desplazamiento físico neto en la dirección solicitada de al menos 0,1 m, sin caída. No se escriben poses raíz durante este ensayo.
6. Solicitar una patada cerca del balón. Deben registrarse una fase de patada original y contacto pie-balón seguido de movimiento del balón de al menos 0,1 m. Se comprueba que sólo se ejecuta una solicitud por evento y que el simulador no inyecta impulso/velocidad al balón.
7. Ejecutar bloqueo central y los dos laterales disponibles; lanzar la pelota desde posiciones iniciales de ensayo controladas. Registrar transición motriz y contacto pelota-portero; no basta mostrar la etiqueta BLOCK. Los casos elegidos deben producir al menos un bloqueo por variante.
8. Ejecutar el flujo automático de penal con ambas instancias B-Human, registrar su resultado y alcanzar finalización/ABORT en el presupuesto activo de 28 s.
9. Repetir tres veces un ensayo con la misma configuración y semilla. Las peticiones y eventos deben conservar su orden, y las poses finales tener diferencia máxima de 1 mm y 0,1°. Si existe una fuente de aleatoriedad adicional, se controla antes de usar la comparación con RL.
10. Ejecutar las pruebas existentes de controlador, visión y empaquetado. Adaptar los tests del simulador a los nuevos contratos físicos; la prueba actual que exige un impulso ideal no valida el backend B-Human.

La comprobación se realizará por etapas: controles de sesión, runtime/protocolo, sensores/postura, marcha, acciones, penal completo. Los registros deberán indicar qué etapa está comprobada; el simulador se declara integrado sólo al cumplir los criterios anteriores.

## Mapa previsto y entrega

- `penalty/simulator.py`: entrada GUI/headless y orquestación del backend físico.
- `penalty/sim_session.py`: estados, reloj, comandos y reinicio del ensayo.
- `penalty/sim_panel.py`: controles Tkinter, atajos y telemetría.
- `penalty/bhuman_bridge.py`: procesos, protocolo y manejo de fallos.
- `penalty/nao_model.py` y `penalty/joint_env.py`: sensores, raíces libres y aplicación de objetivos/rigidez reutilizable.
- `native/bhuman_motion/`: adaptador C++, configuración de proveedores y target de compilación.
- `baseline/bhuman/` y manifiestos: fuentes de origen y dependencias reproducibles.
- `tools/`: migración completa y compilación del runtime.
- `docs/`: inicio, controles, arquitectura, diagnóstico, procedencia y límites observados.

Las fuentes, configuración y documentación se versionan; binarios intermedios y registros voluminosos se excluyen del repositorio. Se conservan las licencias/atribuciones y se actualizan `AGENTS.md`, README y estado de proyecto al completar la integración.

## Fuentes de diseño

- Código local y manifiestos mencionados, revisados en la revisión indicada.
- [Arquitectura de módulos B-Human 2025](https://docs.b-human.de/coderelease2025/architecture/modules-representations/).
- [Framework de movimiento B-Human 2025](https://docs.b-human.de/coderelease2025/motion/motion-framework/).
- [Marcha B-Human 2025](https://docs.b-human.de/coderelease2025/motion/motion-walking/).

La documentación 2025 explica los conceptos; las interfaces y fuentes usadas para compilar se obtienen de la revisión fijada del baseline, evitando mezclar implementaciones de releases distintos.
