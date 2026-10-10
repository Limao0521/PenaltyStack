# Portero robusto frente a distintos estilos

2026-10-10. Continuación de [MEJORA_DE_PORTERO.md](MEJORA_DE_PORTERO.md) en
`codex/mejora-portero`. Objetivo: mejorar el rendimiento equilibrado entre
estilos y el peor estilo, con evaluación reservada en B-Human/MuJoCo.

## Política `robust`

`penalty/rl/robust_keeper.py` implementa `robust_keeper_v1`, una política
táctica de reglas temporales. Conserva `rl_obs_v1`, `keeper_action_v1` y
las recompensas publicadas. B-Human ejecuta todos los movimientos; no se
escriben raíces ni velocidades del balón.

1. Lee solo observaciones finitas, visibles y recientes. Espera ante una
   pelota que se aleja o no presenta avance sostenido hacia el portero.
2. Conserva seis predicciones de postura antes de detectar el avance del
   balón. Dos observaciones consecutivas de avance confirman el tiro.
3. Proyecta el cruce desde posición y velocidad observadas y usa la mediana
   de tres proyecciones para reducir el efecto del ruido.
4. Solicita la guardia central si el cruce queda dentro de ±0.30 m. Cerca
   del borde, una predicción previa de postura con probabilidad ≥0.90 puede
   resolver el lado, siempre que coincida con la dirección del balón.
5. Para una esquina espera hasta que la llegada prevista esté a ≤1.8 s.
   La guardia central puede comenzar antes porque se mantiene; el salto
   lateral tiene una ventana de cobertura finita. El compromiso es único.

La regla geométrica anterior asumía un frenado de 0.6 m/s². En los tiros
inspeccionados la velocidad posterior al golpe varía mucho menos; esa
suposición podía descartar tiros que sí llegaban. La política nueva usa
velocidad constante para la predicción y actualiza el cálculo con nuevas
observaciones. Es un parámetro de esta integración, sujeto a reevaluación
si cambian la pelota, el suelo o el modelo físico.

La política no recibe el objetivo elegido por el rival, su identidad ni el
evento de contacto de MuJoCo. La posición, orientación y velocidad estimada
siguen teniendo origen en la escena; las perturbaciones no las convierten
en percepción real de cámara.

## Selección y prueba reservada

- Selección: semilla base 61001, 12 episodios por estilo, cinco estilos
  (`center`, `mixed`, `corners`, `v1`, `v2`). Se probaron seis combinaciones:
  anticipación temporal de 0.7/1.2/1.8 s y borde central de 0.30/0.40 m.
  La configuración de 1.8 s y 0.30 m se congeló antes de la prueba final.
- Prueba final: semilla base 62001, 100 episodios por estilo, ocho estilos y
  tres porteros. Las mismas semillas se usan para todos los porteros.
- Prueba de percepción: semilla base 63001 y el mismo número de episodios,
  con `keeper_sensor_stress_v2`. Añade incertidumbre de orientación del
  rival (σ=0.15 rad), de velocidad (σ=0.06 m/s), 100 ms de retraso y 10 %
  de pérdida visual. Son perturbaciones analíticas, sin calibración NAO.
- Los parámetros no se reajustan usando ninguna de las dos pruebas finales.

Los estilos adicionales de prueba, `soft`, `late` y `feint`, no se usaron
para seleccionar los parámetros. Los estilos son:

| Estilo | Peticiones tácticas del tirador |
| --- | --- |
| `center` | Centro, pie izquierdo y alcance nominal de 2500 mm |
| `mixed` | Objetivo uniforme en ±0.65 m, ambos pies y potencia variable |
| `corners` | Objetivo entre 0.35 y 0.65 m a ambos lados, ambos pies |
| `soft` | Esquinas con alcance pedido entre 1850 y 2130 mm |
| `late` | Objetivos variados, disparo habilitado entre 6 y 10 s |
| `feint` | Objetivo inicial opuesto; cambia entre 1.5 y 3 s y luego patea |
| `v1` | Política PPO anterior, predominantemente dirigida a un lado |
| `v2` | Política PPO que reparte sus tiros entre las dos esquinas |

Todos comparten el controlador de movimiento B-Human y la aproximación
táctica del proyecto. Esta diversidad no demuestra transferencia a otros
controladores, robots o formas de patear fuera de este sistema.

`center:gate` usa el filtro privilegiado de contacto para comenzar su guardia;
se conserva como referencia de los experimentos anteriores. `anticipation`
y `robust` usan sus observaciones para decidir. El benchmark registra esta
diferencia, las perturbaciones, las semillas y los hashes de procedencia.

## Resultados reservados

100 episodios por estilo y portero; las tasas cuentan los errores como no atajadas.

| Estilo | robust | anticipation | center:gate | robust con percepción degradada |
| --- | --- | --- | --- | --- |
| center | 89 % | 79 % | 89 % | 90 % |
| mixed | 83 % | 64 % | 63 % | 75 % |
| corners | 82 % | 43 % | 42 % | 57 % |
| soft | 86 % | 41 % | 42 % | 58 % |
| late | 77 % | 59 % | 65 % | 75 % |
| feint | 78 % | 46 % | 43 % | 61 % |
| v1 | 92 % | 28 % | 17 % | 73 % |
| v2 | 90 % | 27 % | 1 % | 73 % |

| Condición | Portero | Media entre estilos | Peor estilo |
| --- | --- | --- | --- |
| realista | robust | 84.6 % | 77 % (late) |
| realista | anticipation | 48.4 % | 27 % (v2) |
| realista | center:gate | 45.2 % | 1 % (v2) |
| stress | robust | 70.2 % | 57 % (corners) |
| stress | anticipation | 15.5 % | 1 % (late) |
| stress | center:gate | 41.1 % | 3 % (v2) |

Frente a `anticipation`, la mejora media es de **36.25 puntos** en la condición
original (IC95 pareado: 31.63–40.88) y **54.75 puntos** con percepción degradada
(IC95: 49.50–59.88). La tasa central se mantiene en 89/100, igual a `center:gate`.

En los 2400 episodios de la condición original hubo **cinco `sim_error`**:
dos con robust, dos con anticipation y uno con center. Corresponden a dos
semillas y se reprodujeron como `striker: Nonfinite or out-of-range sensor`.
No se sustituyeron esos episodios. En la condición degradada no hubo errores.

La bandera heredada `early_dive` marca 7/800 compromisos de robust en la
condición original y 26/800 con perturbaciones. Marca compromiso anterior
al contacto etiquetado o con velocidad real inferior a 0.15 m/s; incluye
guardias centrales. No equivale por sí sola a una caída. Queda margen para
mejorar el detector de tiro y su confianza sin reajustar este test reservado.

El resultado cambia el diagnóstico inicial: **el momento de activación y
la estimación de llegada permiten mejorar mucho con las estiradas existentes**.
El oráculo que se comprometía inmediatamente tras el contacto no era un techo
para una política que también decide el momento de bloquear.

## Comparación directa con el portero base del proyecto

El documento inicial llama «FSM de B-Human» a la referencia `scripted`.
Sus decisiones vienen de la FSM del proyecto; B-Human ejecuta los movimientos
originales. Esta comparación no ejecuta toda la pila de comportamiento del
proyecto upstream B-Human. Tanto `scripted` como `robust` usan el mismo motor
de movimiento: cambia la decisión de cuándo bloquear y a qué lado hacerlo.

Se evaluó `scripted` en 800 episodios, con los mismos ocho estilos, las mismas
semillas base 62001 y la misma percepción de la prueba reservada de `robust`.
Se reutilizan los registros congelados de `robust`; no se reajustan parámetros.
El ejecutable, la física, las observaciones y la política robusta conservan
los hashes de la prueba anterior. Los cambios posteriores de código solo
añaden diagnósticos, validación de entradas y estadísticas; el informe los
identifica. Ninguno altera las decisiones ni los resultados de estos episodios.

| Estilo | Base del proyecto (`scripted`) | Nuevo (`robust`) |
| --- | --- | --- |
| Centro | 89/100 | 89/100 |
| Variados (`mixed`) | 63/100 | 83/100 |
| Esquinas | 45/100 | 82/100 |
| Suaves | 45/100 | 86/100 |
| Tardíos | 65/100 | 77/100 |
| Cambio de objetivo (`feint`) | 36/100 | 78/100 |
| Tirador v1 | 20/100 | 92/100 |
| Tirador v2 | 3/100 | 90/100 |
| **Promedio entre estilos** | **45.8 %** | **84.6 %** |

El nuevo ataja aproximadamente **85 de cada 100**, frente a **46 de cada 100**
del base, promediando estos estilos en simulación. La mejora pareada es de
38.875 puntos porcentuales (IC95: 34.00–43.625). Los totales son 677/800 frente
a 366/800. Al centro empatan; la mejora se concentra en los tiros laterales.
El portero base tuvo dos errores de sensores del tirador, en las mismas
combinaciones de estilo y semilla que `robust`; ambos los cuentan como no
atajadas. Los demás resultados incluyen goles, tiros fuera y timeouts: no
todo penal que no se ataja es un gol.

El reporte original del base es `bhuman_base_realista.json`; el análisis
pareado, con procedencia y hashes, es `comparacion_bhuman_base.json`, ambos
en `docs/validation/rl/robustez_20261010/`. La prueba adicional usa la condición
original de percepción; no añade una comparación con `scripted` bajo `stress`.

El caso `center` reproduce las decisiones del tirador base
`ScriptedStrikerPolicy`: centro, pie izquierdo, alcance nominal de 2500 mm
y disparo habilitado desde el inicio. Ambos pasan por el mismo controlador
táctico y el motor B-Human. Los casos `v1` y `v2` usan nuestras políticas
entrenadas. Ninguno ejecuta el comportamiento oficial completo de B-Human.

## Estado en el NAO real y trabajo pendiente

La propuesta de reutilización de visión B-Human, el mapa de fuentes y el
adaptador con medidas desconocidas están en [VISION_PORTERO.md](VISION_PORTERO.md).
La primera etapa ya está implementada como `vision_keeper_v2`: seguimiento
RGB con calibración/odometría, `rl_obs_v2`, adaptador de modelos B-Human y
modo NAOqi de observación. La detección de un rival en las estructuras
importadas no entrega su orientación corporal.

La integración posterior del detector B-Human del rival ya ejecuta la red
superior y el contraste inferior desde imágenes, con posición, edad y
covarianza. El ensayo inicial valida posición en 28/30 capturas; mantiene
orientación desconocida. Los detalles y el destino NAO pendiente de
calibración/compilación están en [VISION_BHUMAN_RIVAL.md](VISION_BHUMAN_RIVAL.md).

La mejora táctica, su configuración y las pruebas están implementadas en
simulación. El portero obtiene la posición del balón y la postura del rival
desde la escena, con las perturbaciones descritas arriba. La postura del
tirador ayuda a resolver casos dudosos cerca del borde de la guardia central;
la trayectoria del balón y su tiempo de llegada guían el bloqueo.

Ya existe un detector básico de pelota naranja en `penalty/vision.py` que
estima distancia y ángulo desde una imagen RGB. Requiere calibración con las
cámaras del NAO concreto. `penalty/nao.py` lee imágenes mediante NAOqi, pero
con el perfil anterior ejecuta la FSM básica. El perfil nuevo
`keeper.vision.example.json` selecciona `vision`, calcula decisiones desde
imágenes y las registra; todavía no ejecuta sus bloqueos físicos.

Para trasladar la mejora al robot falta:

1. Calibrar y validar en el NAO el seguimiento temporal y la compensación
   de cámara/odometría implementados, con imágenes y sensores reales.
2. Estimar la posición y orientación del tirador a partir de imágenes,
   registrando la incertidumbre y los casos en que no se puede medir.
3. Conectar cámaras y sensores a un adaptador de observaciones compatible
   con la política, y conectar sus decisiones al ejecutor físico. Los
   contratos publicados se conservan; cualquier extensión necesita una
   versión nueva. Un único controlador debe ser dueño de los actuadores.
4. Medir los retrasos y la precisión con imágenes grabadas antes de los
   ensayos físicos, y calibrar los bloqueos mediante pruebas incrementales
   en el NAO concreto. Comparar después con el portero base bajo las mismas
   condiciones reales.

La versión `vision` ya usa seguimiento del balón y tiempo de llegada, con
la postura del rival deshabilitada. En cámara simulada hizo 74/80 atajadas
frente a 44/80 del base, en un ensayo preliminar con semillas nuevas;
ver [VISION_PORTERO.md](VISION_PORTERO.md). Incorporar cámaras permite
operar con observaciones reales, pero no garantiza mantener la tasa de
atajadas de la simulación. El **84.6 % no es un resultado en hardware**.

## Métricas y archivos

La media da el mismo peso a cada estilo. El mínimo por estilo permite detectar
especialización. Tiros fuera, caídas, timeouts y errores permanecen en el
denominador y no se cuentan como atajadas. Se guardan todos los episodios;
el resumen incluye IC95 de Wilson por estilo y comparaciones pareadas.
El IC95 de la ganancia media usa bootstrap por semilla, agrupado entre estilos.

Reportes: `docs/validation/rl/robustez_20261010/`. Configuración congelada:
`models/rl/keeper_robust_v1.json`. Se conservan los archivos originales de
recolección, un resumen derivado, el diagnóstico de errores y las fuentes
exactas usadas al recolectar. Los archivos de validación se preservan sin
conversión de finales de línea para mantener sus hashes de procedencia.
Las **82 pruebas de RL y robustez** pasan; el baseline de 559 originales
conserva sus hashes.

## Comandos

```powershell
# Comparación reservada de los ocho estilos; 2400 episodios en total
.\.venv\Scripts\python.exe tools/evaluate_keeper_robustness.py --episodes 100 --seed 62001 --workers 16 --output runs/keeper_robust/heldout_realista.json

# Otra muestra independiente con percepción degradada
.\.venv\Scripts\python.exe tools/evaluate_keeper_robustness.py --condition stress --episodes 100 --seed 63001 --workers 12 --output runs/keeper_robust/heldout_stress.json

# Portero base del proyecto sobre las mismas semillas del nuevo
.\.venv\Scripts\python.exe tools/evaluate_keeper_robustness.py --keepers scripted --episodes 100 --seed 62001 --workers 16 --output runs/keeper_robust/bhuman_base_realista.json

# Ver la política contra el tirador v2
.\.venv\Scripts\python.exe tools/watch_rl.py --striker models/rl/striker_v2 --keeper robust --shots 4 --live

# Pruebas de decisiones temporales, perturbaciones y comparación pareada
.\.venv\Scripts\python.exe -m unittest tests.test_robust_keeper -v
```
