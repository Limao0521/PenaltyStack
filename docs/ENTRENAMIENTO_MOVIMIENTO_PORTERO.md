# Entrenamiento del movimiento del portero

Experimento `keeper_motion_training_v1`, en la rama `codex/mejora-portero`.
Se buscan ajustes reales de los movimientos B-Human en MuJoCo. La base es
el predictor visual v2 con selección `balanced`: se mantiene la misma
predicción, confirmación, defensa elegida y momento de compromiso.

## Qué se aprende

La nueva skill `keeper_motion_skill_v1` modifica una copia de las secuencias
originales de `KeyframeMotionEngine`. B-Human conserva las fases, interpolación,
espejado, balance, rigidez, condiciones de entrada y recuperación.

| Parámetro | Rango | Efecto ejecutado por B-Human |
|---|---:|---|
| `guard_time_scale` | 0.85–1.15 | Multiplica las duraciones activas de guardias centrales y anchas. |
| `jump_time_scale` | 0.85–1.15 | Multiplica la preparación de 130 ms y el alcance de 400 ms del salto. |
| `arm_spread_rad` | −0.08–0.12 rad | Desplaza `ShoulderRoll` de ambos brazos en sentidos opuestos. |
| `guard_knee_offset_rad` | 0–0.10 rad | Añade flexión a ambas rodillas en las guardias; no modifica las rodillas del salto. |

Sólo se limitan los ángulos modificados a `JointLimits`; los valores especiales
`ignore/off` se conservan. Las marcas de fase de 1 ms, la relajación del salto,
las secuencias de levantarse y `prepare` conservan sus valores originales.
El perfil queda fijo desde el primer bloqueo hasta terminar el ensayo.
Una solicitud sin perfil sigue el camino anterior. El perfil nominal
`(1, 1, 0, 0)` conserva exactamente los valores originales, incluso aquellos
que ya estaban fuera de un límite nominal en la fuente.

Los antiguos parámetros de `ParameterizedBlock` continúan sin soporte:
se creó una versión nueva y se conserva su rechazo de ajustes ignorados.
No se reentrena el selector ni se continúa el PPO táctico `keeper_v2`.
Esta primera búsqueda aprende un perfil compartido de cuatro parámetros;
no aprende una política libre que produzca las 26 articulaciones desde cero.

## Diseño del ensayo

1. Dieciséis perfiles fijados antes de entrenar: el nominal, cuatro cambios
   aislados y once combinaciones mediante muestreo estratificado reproducible.
2. Cada perfil ejecuta las seis defensas con raíces libres, servos y contactos
   físicos. Debe completar la defensa y recuperarse con el motor original,
   confirmar postura estable durante 0.5 s y conservarla durante otro segundo
   con apoyo de ambos pies, dentro del límite de 28 s.
3. Entrenamiento: 32 casos, cuatro por cada uno de ocho estilos, con exactamente
   los mismos tiros para los perfiles que pasan la comprobación anterior.
4. Validación: 64 casos nuevos. Se comparan los dos mejores perfiles no nominales
   del entrenamiento y el perfil nominal. Se elige por atajadas medias entre
   estilos, luego por el peor estilo y finalmente por menor cambio del perfil.
5. Se congela el modelo y se prueba contra `balanced` en 160 casos nuevos,
   20 por estilo. El intervalo de ganancia usa bootstrap pareado independiente
   dentro de cada estilo, con 10.000 remuestreos y el mismo peso por estilo.

Las bases de semillas son 81001, 82001 y 83001 para entrenamiento, validación
y prueba. La semilla efectiva es `base*1000000 + indice_estilo*10000 + episodio`.
La prueba se fija en el manifiesto antes de buscar parámetros. Sus resultados
no pueden cambiar el perfil elegido. Los ensayos de integración y postura
usan bases separadas.

El objetivo nuevo `keeper_motion_outcome_v2` es la tasa de atajadas con el
mismo peso por estilo. Fallos, tiros sin compromiso y tiros sin gol permanecen
en el denominador. Un error de simulación invalida la selección. La caída
intencional del salto no se castiga como fallo: la recuperación se comprueba
por separado. Cada comparación exige el mismo reset, contexto del balón,
predicción, defensa solicitada y tiempo de compromiso antes de actuar.

`balanced` normalmente selecciona cuatro defensas. Las seis están habilitadas
y se comprueba su ejecución, pero este experimento no entrena seis perfiles
independientes ni fuerza un uso uniforme de las seis.

## Percepción y alcance

El portero recibe imágenes RGB renderizadas y sus propios encoders, IMU y
FSR simulados. El predictor sólo usa el balón; la orientación del rival sigue
sin intervenir. El tirador recibe la observación privilegiada de escena con
`PerceptionNoise v1`. Los estilos `v1/v2` corresponden a nuestros tiradores
entrenados, no a un equipo B-Human autónomo.

Los sensores motrices son ideales (`mujoco-truth-v2`) y el reloj físico se
pausa durante la inferencia visual. Estos resultados son de simulación.
No se ejecutaron movimientos en un NAO físico. No se escriben trayectorias
de raíz ni velocidades del balón para conseguir los bloqueos; los estados
iniciales se colocan únicamente al reiniciar el ensayo.

## Reproducción

```powershell
.venv/Scripts/python.exe tools/build_motion_runtime.py
.venv/Scripts/python.exe -m unittest tests.test_keeper_motion tests.test_native_keeper_motion tests.test_motion_training -v
.venv/Scripts/python.exe tools/train_keeper_motion.py --output runs/keeper_motion/nueva_corrida --model-output models/vision/keeper_motion_nueva.json --workers 6
```

Las etapas `safety`, `search`, `fit` y `test` también se pueden ejecutar por
separado. Para continuar una misma corrida, añadir `--resume`; deben coincidir
los parámetros del ensayo, versiones, fuentes, modelos y ejecutables.
El modelo congelado no se sobrescribe con otro perfil.

```powershell
.venv/Scripts/python.exe tools/watch_keeper_motion.py --model models/vision/keeper_motion_v1.json --style mixed --seed 84001000000
.venv/Scripts/python.exe tools/validate_keeper_motion.py --output runs/keeper_motion/posturas --render
```

La validación con `--parameters <JSON>` ejecuta un perfil concreto; `--legacy`
prueba la solicitud anterior. `--render` produce imágenes por defensa.

## Resultado de la primera corrida

La búsqueda probó 16 perfiles en 512 duelos de entrenamiento. La validación
eligió `candidate_02`: `jump_time_scale=0.85` y los demás parámetros nominales.
En la ruta usada del salto, las fases de 130 y 400 ms pasan a 110.5 y 340 ms.
Los ángulos finales de brazos y rodillas conservan los valores originales.
Esto acorta esas fases un 15 %; no significa que toda la reacción sea un 15 %
más rápida.

| Etapa | Casos nuevos | Movimiento original | Perfil elegido |
|---|---:|---:|---:|
| Entrenamiento | 32 | 29/32 | 27/32 |
| Validación | 64 | 54/64 | 56/64 |
| Prueba reservada | 160 | 134/160 (83.75 %) | 136/160 (85.00 %) |

En la prueba reservada ganó cinco casos y perdió tres: ganancia neta de
1.25 puntos porcentuales, IC95 pareado **−1.875 a +5.0 puntos**. Frente a
nuestro tirador v2 perdió una atajada (15/20 frente a 16/20). No hay evidencia
suficiente para declarar una mejora general. El perfil queda **experimental**;
el portero predeterminado conserva su configuración.

Las 96 comprobaciones de ejecución/recuperación y las 305 pruebas del proyecto
pasaron. La predicción, defensa elegida y tiempo de compromiso coincidieron
en todos los pares; cero errores de simulación. Las seis defensas están
habilitadas, pero `balanced` eligió cuatro en este ensayo.

Modelo congelado: [`keeper_motion_v1.json`](../models/vision/keeper_motion_v1.json).
Resultados, trazas, auditorías y ejemplo visual de un perfil descartado:
[`keeper_motion_v1`](validation/vision/keeper_motion_v1/README.md).
