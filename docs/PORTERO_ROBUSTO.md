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

# Ver la política contra el tirador v2
.\.venv\Scripts\python.exe tools/watch_rl.py --striker models/rl/striker_v2 --keeper robust --shots 4 --live

# Pruebas de decisiones temporales, perturbaciones y comparación pareada
.\.venv\Scripts\python.exe -m unittest tests.test_robust_keeper -v
```
