# Entrenamiento de seis defensas sobre el portero visual v2

El punto de partida es `vision_keeper_v2`, el portero de reglas que predice el cruce del balón desde cámara. El checkpoint PPO antiguo `models/rl/keeper_v2` es otro modelo y no se continúa: su informe indica compromisos antes del tiro y caídas. Se publica un modelo nuevo, `vision_keeper_selector_v1`, conservando v2 como referencia.

## Qué se aprende

La nueva política aprende **cuál de las seis defensas originales B-Human solicitar** cuando el núcleo v2 decide comprometerse. No aprende trayectorias articulares, no altera la predicción del balón y no modifica el momento de activación. Las seis opciones son elegibles incluso cuando v2 había clasificado otro lado. La preparación queda fuera de estas acciones.

La predicción temporal sigue siendo v2: dos capturas entrantes, mediana de hasta tres proyecciones, zona central entre ±0.30 m, centro al confirmar y saltos cuando la llegada prevista es ≤1.8 s. La primera defensa seleccionada permanece fija; se conserva el límite activo de 28 s.

El instante conservado depende de la clasificación de v2. Elegir un salto en un compromiso originalmente central puede ejecutarlo antes de lo habitual; elegir una guardia central en un compromiso lateral puede ejecutarla más tarde. Estos efectos forman parte del resultado que aprende el selector. Este experimento no optimiza cuándo iniciar cada plantilla.

## Datos disponibles para decidir

Contrato nuevo `keeper_defense_context_v1`: cruce previsto, tiempo de llegada, posición x/y y velocidad x/y del balón, edad de medida y varianzas de velocidad x/y, capturados justo antes del primer compromiso. Provienen de RGB y sensores propios simulados. El modelo no recibe estilo, semilla, comandos del tirador, objetivo real, resultados futuros ni yaw del rival.

Para una decisión por penal se ajustan seis estimaciones de atajada a partir de los resultados de las seis acciones. Es un entrenamiento contextual con resultados contrafactuales medidos en física. Evita tener que reaprender la espera y la trayectoria que v2 ya resuelve. No se continúa el PPO de acción continua y observación privilegiada publicado anteriormente.

## Recolección y objetivo

Para cada caso de estilo/semilla se repite el episodio completo seis veces, con una defensa diferente al mismo instante. Se utilizan raíces libres, servos, contactos y el motor original B-Human. No se copia sólo el estado MuJoCo para ramificar un penal: el motor nativo mantiene estado interno que también tendría que restaurarse.

Antes de considerar un grupo apto para entrenamiento, se comprueba que las seis repeticiones compartan el contexto y el compromiso de v2. Se registra por separado si no hubo compromiso o si el simulador falló. Un resultado de infraestructura no se convierte en una etiqueta física.

Objetivo nuevo `keeper_defense_outcome_v1`: atajada = 1; los otros resultados físicos = 0. Se ajustan todas las opciones, sin inventar una etiqueta única cuando varias defensas atajan. Los errores se excluyen del ajuste y permanecen en los reportes. Los casos sin compromiso no enseñan a seleccionar, pero siguen en el denominador de evaluación.

El modelo contiene seis regresiones logísticas: nueve campos normalizados y tres términos cuadráticos del cruce/llegada, ajustados con BCE, regularización L2 y Adam de NumPy. Son puntuaciones experimentales de atajada, sin calibración de probabilidades validada. La normalización usa exclusivamente entrenamiento. La regularización y el margen de desempate hacia la defensa original se seleccionan con validación. La prueba final evalúa el modelo congelado; sus resultados no reajustan pesos ni parámetros.

## Comparación independiente

Tres particiones de semillas disjuntas:

| Partición | Base de semilla | Uso |
| --- | --- | --- |
| Entrenamiento | 74001 | Ajustar seis puntuaciones |
| Validación | 75001 | Elegir regularización y margen |
| Prueba | 76001 | Comparar el modelo congelado |

Dentro de cada partición la semilla real incorpora el índice del estilo y del episodio. Cada candidato recibe exactamente el mismo caso físico dentro de un estilo. La comparación confirmatoria remuestrea pares de resultados independientemente dentro de cada estilo y da el mismo peso a los ocho estilos. El reporte automático previo agrupa índices entre estilos; ambos métodos y las semillas reales permanecen explícitos en sus reportes.

El ensayo de integración corto utilizó el índice 0 de prueba para `center` y `mixed`. **Antes del ajuste del modelo y de la prueba de producción** se reservó la comparación confirmatoria para los índices **1 a 20** de cada estilo: se excluye el índice 0 de los ocho estilos y se añade el 20. Se conserva el reporte automático original, y el reporte independiente reutiliza 152 casos y ejecuta ocho adicionales. El protocolo queda fijado en `heldout_protocol.json`; no se eligieron casos en función de resultados.

La prueba compara el selector con v2 original y con `balanced` de la biblioteca v6. Reporta atajadas totales y por estilo, compromisos, frecuencia de cada defensa y errores. La mejor defensa observada retrospectivamente entre seis sólo es una cota diagnóstica; no es un portero ejecutable sin conocer el resultado futuro.

Una primera corrida permite evaluar la idea. Si el margen de mejora incluye cero o falla frente a un estilo, no se promueve el selector aprendido como mejora confirmada; v2 permanece como núcleo y control. Seis opciones disponibles no garantizan mayor eficacia.

La puntuación offline utilizada para elegir regularización no equivale a una tasa final física: los grupos sin contexto válido reciben puntuación cero, incluso si el episodio terminó en atajada sin compromiso. La prueba final conserva el resultado físico real de esos casos y todos los errores en el denominador.

## Resultado de la primera corrida

Entrenamiento: 96 casos, 91 contextos elegibles y 576 duelos. Validación: 48 casos, 43 contextos elegibles y 288 duelos. Cero errores o discordancias; cada defensa recibió 91 etiquetas de entrenamiento y 43 de validación. Se eligieron L2 = 0.1 y margen = 0.03 exclusivamente con validación.

La prueba ciega usa los mismos 160 tiros nuevos para cada política:

| Política | Atajadas | Tasa | Peor estilo |
| --- | ---: | ---: | ---: |
| V2 original | 128/160 | 80.000 % | 65 % |
| V2 con centro según el lado (`balanced`) | 134/160 | 83.750 % | 75 % |
| V2 con selector aprendido de seis opciones | 135/160 | 84.375 % | 75 % |

El selector gana 4.375 puntos frente a v2 (IC95: 0 a 9.375) y 0.625 frente a `balanced` (IC95: −3.125 a 4.375). No confirma una mejora adicional del entrenamiento sobre la regla simple. `balanced` gana seis casos y pierde cero frente a v2; su intervalo es +1.25 a +6.875 puntos. El selector pierde una atajada central, mejora esquinas y empata frente a nuestro tirador v2.

Las seis opciones están habilitadas, pero seleccionó cuatro: centro espejado 53, ancha izquierda 1, salto izquierdo 67 y salto derecho 27; hubo 12 casos sin compromiso. Centro original y ancha derecha fueron elegibles y entrenadas, pero no se eligieron en estos tiros. Se conserva el modelo experimental, sin reemplazar el predictor ni los valores predeterminados publicados. La alternativa sencilla para seguir evaluando en simulación es v2 con `balanced`.

Modelo: `models/vision/keeper_selector_v1.json`. Evidencia, intervalos, datos, comprobaciones y trazas en [keeper_selector_v1](validation/vision/keeper_selector_v1/README.md). Suite completa previa al módulo ciego: 252 pruebas; 14 pruebas adicionales del protocolo pasan. Dos trazas reproducen 38 decisiones exactas y conservan la predicción y activación de v2. Todo corresponde a simulación.

## Procedencia y alcance

Cada partición y checkpoint registra versiones, fuentes, runtime, modelos de tirador y semillas para reanudar sin mezclar ejecuciones incompatibles. No se editan contratos, recompensas, observaciones ni modelos publicados. La ampliación de acciones se describe en [DEFENSAS_PORTERO.md](DEFENSAS_PORTERO.md).

La percepción utiliza el pipeline de cámaras y encoders/IMU/FSR propios. Los sensores simulados son ideales y la inferencia pausa el reloj físico de MuJoCo; el motor conserva el perfil `mujoco-truth-v2`. Los tiradores usan la observación privilegiada con ruido ya declarada en el benchmark. Los estilos `v1` y `v2` son nuestros checkpoints de tirador con control motor B-Human; no representan el comportamiento autónomo completo del equipo B-Human. No se mide hardware ni su latencia.

## Ejecutar y reanudar

```powershell
# Corrida nueva: recolectar y fijar la prueba antes de ajustar pesos.
# Usar una carpeta y un destino de modelo nuevos para no sobrescribir artefactos.
.venv/Scripts/python.exe tools/train_keeper_selector.py --stage collect --workers 6 --output runs/keeper_selector/nueva --model-output runs/keeper_selector/nueva/model.json
.venv/Scripts/python.exe tools/evaluate_keeper_selector_heldout.py --init-protocol --output runs/keeper_selector/nueva

# Ajustar con entrenamiento/validación, congelar y comparar.
.venv/Scripts/python.exe tools/train_keeper_selector.py --stage fit --workers 6 --output runs/keeper_selector/nueva --model-output runs/keeper_selector/nueva/model.json --resume
.venv/Scripts/python.exe tools/train_keeper_selector.py --stage evaluate --workers 6 --output runs/keeper_selector/nueva --model-output runs/keeper_selector/nueva/model.json --resume
.venv/Scripts/python.exe tools/evaluate_keeper_selector_heldout.py --output runs/keeper_selector/nueva --model runs/keeper_selector/nueva/model.json --workers 6

# Reanudar la corrida publicada sin mezclar fuentes/versiones diferentes.
.venv/Scripts/python.exe tools/train_keeper_selector.py --workers 6 --output runs/keeper_selector/v1 --resume

# Ver el selector entrenado en un penal con sensores/cámaras simulados.
.venv/Scripts/python.exe tools/watch_keeper_selector.py --style mixed --seed 77001000000 --live
```

Son 96 casos de entrenamiento y 48 de validación, cada uno repetido con seis defensas; la prueba confirmatoria compara tres políticas en 160 casos nuevos. La corrida automática ejecuta 1344 duelos y la reserva independiente añade 24: 1368 ejecuciones físicas, además del ensayo corto de integración. El modelo se guarda aparte en `models/vision/keeper_selector_v1.json`; los resultados de la corrida se registran junto con sus hashes y no sobrescriben los informes anteriores.

This product includes software developed by B-Human (http://www.b-human.de).
