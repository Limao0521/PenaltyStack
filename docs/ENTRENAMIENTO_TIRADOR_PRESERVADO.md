# Reentrenamiento del tirador preservando v2

El segundo ensayo parte del actor de `models/rl/striker_v2` sin reducir sus
salidas ni cambiar sus decisiones deterministas. El ensayo anterior,
`striker_visual_v3`, se conserva completo: su fallo se recuperó desde un
checkpoint y la pérdida de rendimiento ya aparecía en la inicialización.
No se atribuye toda esa pérdida al fallo de simulación.

## Inicialización y exploración

La nueva política `striker_preserving_gaussian_v1` conserva byte a byte los
pesos del actor, incluida su cabeza de acciones. Para muestrear durante PPO,
comprime únicamente las colas de las medias gaussianas:

```text
C(m) = m                         si -1 <= m <= 1
C(m) = 1 + 0.01 * (m - 1)        si m > 1
C(m) = -1 + 0.01 * (m + 1)       si m < -1
```

La identidad `clip(C(m), -1, 1) == clip(m, -1, 1)` conserva exactamente las
acciones deterministas iniciales de v2 para cualquier media finita. La
exploración deja de concentrarse alrededor de medias saturadas muy alejadas
del rango físico. El gradiente fuera del rango es el de esta función, 0.01;
no se sustituye artificialmente por uno. Muestreo y cálculo de probabilidades
usan la misma distribución gaussiana nueva.

La inicialización `striker_policy_init_v3` conserva la normalización de
observaciones de v2. Reinicia Adam para el objetivo nuevo, pone la salida
final del crítico en cero y fija sigma inicial en 0.25. Esos cambios no
alteran las acciones deterministas. Sigma se limita a [0.12, 0.40].

La exportación NumPy habitual sigue siendo compatible para inferencia
determinista, porque recorta las mismas medias crudas. Su muestreo gaussiano
no implementa esta compresión: este experimento se entrena y reanuda con
`tools/retrain_striker_v2.py`, mediante PPO síncrono con la política nueva.

## Objetivo y física

La recompensa nueva `striker_visual_reward_v3` da cero por paso, +1 por gol
y -1 por cualquier otro resultado terminal físico, incluido timeout. Con
gamma=1 y el horizonte observado de 28 segundos, el retorno esperado es
`2 * P(gol) - 1`. Una derrota tardía no obtiene ventaja por descuento y un
gol tardío no pierde su recompensa por esperar. Un error de simulación es
un diagnóstico fatal: se descarta el rollout incompleto antes de optimizar.

Se conservan los contratos publicados `rl_obs_v1` de 19 campos y
`striker_action_v1` de cuatro decisiones: objetivo, pie, potencia y activación.
B-Human ejecuta marcha y patada; no se aprenden trayectorias articulares.
La observación del tirador sigue siendo información privilegiada de escena
con `PerceptionNoise v1`. El portero visual usa RGB y sensores propios
ideales simulados. No es una prueba en hardware.

## Protocolo fijado antes del aprendizaje

- 96 actualizaciones de 128 pasos por seis entornos: 73 728 transiciones
  optimizadas nuevas como presupuesto, con límite de dos horas de entrenamiento.
  No se afirma convergencia por completar este presupuesto.
- Cinco entornos entrenan contra nuestro portero visual v2 `balanced`, con
  movimientos originales; uno contra nuestra FSM base con motor B-Human,
  para conservar la capacidad anterior. La FSM no es el comportamiento
  oficial completo de B-Human.
- PPO: tasa 0.00003, cuatro épocas, clip 0.05, KL objetivo 0.01, gamma 1,
  GAE 0.95, sin recompensa normalizada ni bono de entropía.
- Semillas físicas nuevas: `101001 * 1000000 + i` para entrenamiento,
  `102001 * 1000000 + i` para validación y `103001 * 1000000 + i` para prueba.
  Las comprobaciones de paridad usan un espacio separado, 104001.
- Validación: 64 casos pareados por política y rival, contra `balanced` y
  la FSM base. Se comparan v2, la inicialización y los checkpoints 32, 64 y 96.
  La selección favorece goles ante `balanced` entre candidatos que no pierden
  más de diez puntos de goles frente a v2 ante la FSM. Es un umbral observado
  de preservación, no una demostración estadística de no inferioridad.
- La promoción se decide por separado: debe superar a v2 en validación,
  cumplir preservación y obtener un intervalo de ganancia de goles con
  límite inferior positivo ante `balanced` en prueba, sin perder más de
  diez puntos ante la FSM en esa muestra.
- Prueba: 100 penales por tirador y rival, 600 en total. Rivales: `balanced`,
  `balanced` con salto ajustado y FSM base. La selección queda congelada
  antes de la prueba; sus resultados no reajustan los pesos ni el protocolo.

Las parejas comparten semillas y condiciones iniciales dentro de cada rival.
Se cuentan goles, atajadas, tiros fuera/cortos, caídas y timeout por separado.
Los intervalos de ganancia usan bootstrap pareado; los secundarios son
descriptivos y no tienen corrección por comparaciones múltiples.

## Fallos y evidencia

Cada fallo conserva inmediatamente semilla, comando, mensaje y evidencia
disponible, antes de reiniciar o recoger otros episodios. Las excepciones
propagadas conservan su traceback. Si el backend devuelve un error cuyo
traceback ya descartó, se etiqueta esa limitación y se guarda la pila del
punto de llamada por separado. No se inventa el traceback original.

Se reservan antes de cada rollout los posibles índices de semillas para
evitar reutilizarlos al reanudar después de una interrupción. Checkpoints
completos conservan pesos, Adam y RNG; la reanudación empieza episodios
físicos nuevos y no pretende restaurar un estado exacto de MuJoCo.

La corrida guarda versiones y hashes de entradas, runtime, modelos,
normalización, evaluaciones y checkpoints. Los resultados están en el
[reporte preserving_v4](validation/rl/striker_preserving_v4/README.md);
el modelo anterior y sus reportes se conservan sin editar.

## Ejecutar con los modelos publicados

Los pesos ya entrenados están en `models/rl/`. Cada modelo incluye
`model.zip`, `obs_norm.npz` y `config.json`; la recomendación y el estado
experimental se consultan en `models/rl/manifest.json`.

Desde una copia preparada del repositorio, estos comandos muestran el
tirador nuevo frente a la FSM base, o el portero visual frente al tirador v2:

```powershell
.\.venv\Scripts\python.exe tools/watch_rl.py --striker models/rl/striker_preserving_v4 --keeper scripted --shots 4 --live
.\.venv\Scripts\python.exe tools/watch_keeper_defenses.py --style v2 --mode balanced --live
```

Para entrenar otra corrida, usa una carpeta nueva. El runner parte de
`striker_v2`, entrena, selecciona por validación y ejecuta la prueba reservada:

```powershell
.\.venv\Scripts\python.exe tools/retrain_striker_v2.py --output runs/striker_retrain_v2/nueva_corrida
```

La preparación de dependencias y runtimes de una copia nueva se explica en
[RL.md](RL.md#instalación). El ZIP del reporte conserva los checkpoints y
los inputs de la corrida anterior; sus binarios Windows están ligados al
entorno de compilación original. En otra ubicación o equipo, compila los
runtimes para esa copia antes de ejecutar nuevos penales.
