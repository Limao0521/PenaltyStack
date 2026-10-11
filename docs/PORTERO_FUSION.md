# Orientación y balón en el portero: revisión, v4 y v5

2026-10-10, rama `codex/mejora-portero`. Continúa
[ORIENTACION_PORTERO.md](ORIENTACION_PORTERO.md). Todo ocurre en simulación
(MuJoCo + motor B-Human original, cámaras RGB y sensores propios simulados).
No hay validación en hardware.

## Resumen

- **v3 nunca usaba la orientación.** La postura del tirador sólo se acepta si
  se midió hasta 0.5 s antes del compromiso. Además, en el cuadro del compromiso
  debe haber yaw válido. Durante la patada el cuerpo deja de ser medible: en
  200 penales de desarrollo, la última postura válida quedó entre 0.7 y 2.8 s
  antes del compromiso (mediana 1.1 s). Las condiciones de v3 no se cumplieron
  en ningún caso, y v3 decide igual que v2.
- **La orientación aporta poca información sobre el cruce.** La línea del torso
  medida por visión predice el cruce con un error de ~0.15–0.22 m. El balón
  queda a ~3 cm tras dos capturas y a <1 cm cuando se comprometen los lados.
  La dirección del tiro depende del pie que patea: +12 cm con uno y −15 cm con
  el otro respecto a la línea del torso. El estimador de yaw no es el problema:
  su predicción difiere 4.6 cm rms de la obtenida con el yaw verdadero.
- **v4** corrige el uso de la orientación sobre la lógica de balón de v2.
  Congela la postura al salir el balón de su reposo, la fusiona con su
  incertidumbre calibrada y retiene una guardia central contradicha por la
  postura. Con el modelo ajustado reproduce v2 en los 182 penales de desarrollo.
- **La guardia central original es asimétrica y su momento importa.** El
  proyecto pide siempre `squatArmsBackLeft`. En la frontera (|cruce|
  0.18–0.36 m), al lado izquierdo del portero el centro ataja 28/34 y la
  estirada 18/34. Al lado derecho, un centro pedido al instante (como v2)
  ataja 7/30. Si se pide a 1.8 s de la llegada, ataja 17/30.
- **v5** cambia la lógica del balón:
  - recta por mínimos cuadrados sobre la trayectoria entrante;
  - el centro también espera a 1.8 s de la llegada;
  - límites por lado: centro hasta 0.35 m a la izquierda y 0.30 m a la derecha.

  `ball_v5` es la misma política sin orientación.
- **Experimento final pareado** (Windows, semillas nuevas 70001, 50 penales
  por estilo y portero, ocho estilos):
  - FSM base: 35.2 %.
  - v2: 74.8 % (+39.5 pp frente a la FSM, IC95 +31.8 a +47.2).
  - v3: 75.0 %. Nunca usó la postura (0/400).
  - v4: 75.0 %. Tuvo postura en 379/400 penales y cambió 1 decisión.
  - v5 balón: 73.5 % (−1.2 pp frente a v2, IC95 −5.8 a +3.0).
  - v5 con orientación: 73.8 %.

  **La orientación no aporta una mejora medible**, y v5 tampoco mejora a v2:
  gana en la frontera derecha y en esquinas lejanas, pero pierde tiros
  centrales por pedir el centro tarde. **Recomendación: mantener v2** (o v4,
  que decide lo mismo).

## Revisión de lo agregado para la visión del portero

| Commit | Qué agrega |
| --- | --- |
| `988d278` | Balón por cámara: detector naranja, proyección calibrada, seguimiento con odometría y covarianza; `vision_keeper_v2`; observador NAOqi |
| `0c10de5` | Rival con el `RobotDetector` B-Human (ONNX superior, contraste inferior) en proceso nativo; proyección de pies y asociación al balón |
| `6e29b46` | Yaw corporal por comparación con 2294 referencias; encoders/IMU/FSR para cámara y odometría; `vision_keeper_v3` |

Puntos sólidos: la verdad sólo entra en campos de evaluación; los contratos
están versionados; las trazas se pueden reproducir; los procesos nativos se
cierran ante fallos. El estimador de yaw es preciso cuando la medida es válida.

Hallazgos, de mayor a menor impacto:

1. **v3 no puede usar la postura** (descrito arriba). Además, sólo puede cambiar
   la decisión dentro de una banda de 4 cm alrededor de 0.30 m y sólo de
   centro a lado. El predictor de tres clases se entrenó con poses ideales y
   frontera 0.20 m, con umbral 0.90. La prueba unitaria pasa porque sus cuadros
   sintéticos están a 0.1 s del compromiso.
2. **v2 compromete el centro con dos capturas.** Los lados esperan 1.8 s, pero
   el centro se decide al segundo cuadro entrante, con la mediana de
   proyecciones de dos puntos (error rms 3.4 cm, máximo 16 cm). En la frontera
   eso produce decisiones del lado equivocado: 10/178 en desarrollo, frente a
   4/178 de v5.
3. **Inicio tardío en tiros lentos.** El umbral de velocidad puede tardar varios
   segundos tras el contacto (ejemplo: contacto 3.7 s, detección 7.6 s). En
   v3, posturas posteriores a la patada podrían entrar en la ventana. v4/v5
   congelan la postura cuando el balón se aleja 6 cm de su reposo.
4. **Poca evidencia estadística previa.** La comparación v3/v2/FSM tenía un
   penal por estilo (8 por portero).
5. **Riesgos para hardware**, sin efecto en simulación:
   - Las marcas de tiempo de NAOqi se convierten con el reloj de pared; si su
     base es monótona, el observador se detiene por «future camera timestamp».
   - Una diferencia de 20 ms entre relojes detiene la observación.
   - Las referencias de yaw se renderizaron con la cámara inferior, pero también
     se aplican a recortes de la superior.
   - El piso de varianza de 7.5° es ~4 veces el error medido.

## Diseño de v4 y v5

`penalty/keeper_fusion.py`, entradas `vision_state_v2` únicamente.

- **Postura previa.** Se guardan medidas de yaw válidas, recientes y con
  varianza ≤(15°)². Cada una da `pointing_y`, el punto donde la línea del torso
  que pasa por el balón corta la línea de gol. La previa usa la mediana de las
  últimas cinco medidas (2.5 s) antes de que el balón deje su reposo. El modelo
  `models/vision/kick_heading_v1.json` la convierte en cruce con σ calibrada.
- **v4 = v2 + postura.** La proyección del balón (σ según el número de capturas)
  se fusiona con la previa. Una guardia central se retiene si la previa apunta
  a una esquina y contradice al balón por más de 2σ combinadas, mientras falten
  más de 1.8 s. Sin previa válida, cada acción es la de v2 (prueba unitaria
  acción por acción).
- **v5.** Recta por mínimos cuadrados sobre las últimas 10 capturas entrantes.
  La llegada se calcula con la velocidad ajustada y σ con los residuos (piso
  2 cm). Centro y lados se comprometen cuando la llegada está a ≤1.8 s. El
  centro cubre hasta 0.35 m a la izquierda y 0.30 m a la derecha. Si el balón
  se aleja o se pierde >0.35 s, la trayectoria se reinicia. El compromiso
  inmediato de un centro claro queda como opción (`centre_confidence`),
  desactivada.

## Desarrollo (semillas 69001, separadas del test)

- Observador que nunca se compromete: 200 penales, ocho estilos, trazas por
  cuadro. 182 con cruce libre medible (verdad sólo para etiquetas). Hay previa
  de postura en 170/182.
- `tools/fit_kick_heading.py`: ajuste robusto `cruce ≈ 0.029 + 1.20·pointing_y`.
  σ dejando un estilo fuera: 0.22 m (rms 0.18 m).
- Repetición offline de decisiones (sin simular): v4 coincide con v2 en las
  182; v5 cambia 8 decisiones respecto a v2, 7 hacia la zona correcta.
- Sonda de acciones en la frontera: 64 semillas con |cruce| 0.18–0.36 m. Cada
  semilla se jugó con estirada al lado, centro al instante y centro a 1.8 s.

| Lado del cruce | |cruce| (m) | Estirada | Centro al instante | Centro a 1.8 s |
| --- | --- | --- | --- | --- |
| izquierda (+y) | 0.18–0.25 | 15/18 | 15/18 | 13/18 |
| izquierda (+y) | 0.25–0.30 | 2/12 | 9/12 | 9/12 |
| izquierda (+y) | 0.30–0.36 | 1/4 | 4/4 | 4/4 |
| derecha (−y) | 0.18–0.25 | 16/18 | 5/18 | 12/18 |
| derecha (−y) | 0.25–0.30 | 1/5 | 1/5 | 4/5 |
| derecha (−y) | 0.30–0.36 | 2/7 | 1/7 | 1/7 |

  Las muestras son pequeñas. La estirada falla entre 0.25 y 0.36 m a ambos
  lados; el centro tardío es la mejor opción en esa franja. Los límites de v5
  se fijaron con esta tabla antes del test final.
- Control con el estilo `center` (25 semillas, casi todo |cruce| <0.25 m):
  v2 ataja 23/25 y el centro tardío de v5 22/25 (2 pierde, 1 gana).
  Esperar no daña los tiros centrales en esta muestra.

Las corridas de desarrollo usaron una compilación Linux propia del mismo código
(host de movimiento y detector), sólo para iterar. El test final se ejecuta en
Windows con los ejecutables ya validados.

## Experimento final

Reporte, análisis, logs y pruebas en
[validation/vision/fusion_final_20261011](validation/vision/fusion_final_20261011).
2026-10-11, Windows (Python 3.11, ejecutables MSVC validados), 16 workers.
Las semillas 70001 no se usaron en el desarrollo. Los seis porteros enfrentan
los mismos tiros (comparación pareada). La suite pasó (198 pruebas). Los
`sim_error` son tiempos de espera del detector nativo (0.25 s de reloj real)
y cuentan como no atajadas: 23 de la FSM, casi todos al arrancar los 16 procesos,
1 de v2 y 1 de v4. Sin ellos, la FSM queda en 37.4 % (141/377).

| Portero | center | mixed | corners | soft | late | feint | v1 | v2 | Media | Peor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FSM base | 40/50 | 19/50 | 13/50 | 17/50 | 27/50 | 14/50 | 8/50 | 3/50 | 35.2 % | v2 6 % |
| v2 (sin orientación) | 43/50 | 32/50 | 39/50 | 37/50 | 38/50 | 36/50 | 38/50 | 36/50 | 74.8 % | mixed 64 % |
| v3 (orientación actual) | 43/50 | 32/50 | 39/50 | 38/50 | 38/50 | 36/50 | 38/50 | 36/50 | 75.0 % | mixed 64 % |
| v4 (v2 + orientación) | 43/50 | 32/50 | 39/50 | 38/50 | 38/50 | 36/50 | 38/50 | 36/50 | 75.0 % | mixed 64 % |
| v5 balón | 37/50 | 37/50 | 38/50 | 41/50 | 30/50 | 35/50 | 39/50 | 37/50 | 73.5 % | late 60 % |
| v5 balón + orientación | 37/50 | 38/50 | 38/50 | 41/50 | 30/50 | 35/50 | 39/50 | 37/50 | 73.8 % | late 60 % |

Comparaciones pareadas (bootstrap por semilla; «solo A ataja / solo B ataja»):

| Comparación | Diferencia media | IC95 | Discordantes |
| --- | --- | --- | --- |
| v2 − FSM base | +39.5 pp | +31.8 a +47.2 | 179 / 21 |
| v3 − v2 | +0.2 pp | 0.0 a +0.8 | 1 / 0 |
| v4 − v2 | +0.2 pp | 0.0 a +0.8 | 1 / 0 |
| v5 balón − v2 | −1.2 pp | −5.8 a +3.0 | 34 / 39 |
| v5 − v5 balón (orientación) | +0.2 pp | 0.0 a +0.8 | 1 / 0 |
| v5 − FSM base | +38.5 pp | +30.2 a +46.5 | 188 / 34 |

v3 coincide en las decisiones válidas con v2. v4 cambia una decisión de lado
y momento en `feint`, semilla `70001000007`: centro a 10.800 s frente a derecha
a 11.808 s; ambas reciben gol. La atajada adicional de v3/v4 corresponde a un
tiempo de espera del detector en un penal de v2. En v4 la postura estuvo
disponible en 379/400 penales, cambió una decisión y no retuvo ningún centro.
El modelo de orientación, verificado en las trazas del observador de estas
semillas, tiene residuo rms de 0.153 m (σ robusta 0.18 m, p90 0.225 m), igual
que en desarrollo.

Atajadas por cruce libre real (observador en las mismas semillas, 350 tiros):

| Lado y |cruce| (m) | FSM base | v2 | v5 balón |
| --- | --- | --- | --- |
| izquierda 0–0.15 | 25/31 | 28/31 | 26/31 |
| izquierda 0.15–0.25 | 31/43 | 34/43 | 27/43 |
| izquierda 0.25–0.36 | 22/31 | 23/31 | 22/31 |
| izquierda 0.36–0.60 | 11/92 | 82/92 | 83/92 |
| izquierda >0.60 | 0/22 | 8/22 | 11/22 |
| derecha 0–0.15 | 18/20 | 20/20 | 14/20 |
| derecha 0.15–0.25 | 10/29 | 15/29 | 15/29 |
| derecha 0.25–0.36 | 0/26 | 7/26 | 13/26 |
| derecha 0.36–0.60 | 1/53 | 51/53 | 50/53 |
| derecha >0.60 | 0/3 | 2/3 | 2/3 |

v5 pide el centro con mediana 1.76 s antes de la llegada en tiros centrales.
v2 lo pide con mediana 2.68 s. Esperar pierde 15 atajadas con |cruce| <0.25 m,
y las ganancias en la frontera derecha (+6) y las esquinas lejanas (+3) no lo
compensan. La sonda de desarrollo incluyó 25 tiros centrales con centro tardío,
pero la muestra no anticipó la regresión final. La mejora de la frontera
derecha sí se confirma.

### Conclusiones

1. La percepción de v2 ya resuelve el lado con la trayectoria del balón. La
   orientación cambia alguna decisión, pero no produce una mejora general
   demostrada en este experimento.
   Para que importara habría que predecir el pie de golpeo (±12–15 cm de
   desvío) y, aun así, el residuo (5–9 cm) seguiría siendo peor que el balón
   tras dos capturas.
2. El límite actual está en los movimientos:
   - la guardia central `squatArmsBackLeft` cubre mal el lado derecho;
   - la estirada falla entre 0.25 y 0.36 m;
   - los tiros a más de 0.6 m superan la estirada.
3. Siguientes pasos, con semillas nuevas:
   - usar el centro al instante para tiros centrales y el centro tardío o la
     estirada sólo en la frontera derecha;
   - probar `squatArmsBackRight`/`squatWideArmsBack*` para el lado derecho
     (requiere cambiar y recompilar el host de movimiento).
4. Todo es simulación. Faltan la cámara real, la calibración y la validación
   en el NAO.

## Reproducir

Ampliación posterior del catálogo de movimientos y explicación del predictor:
[DEFENSAS_PORTERO.md](DEFENSAS_PORTERO.md). Los resultados anteriores se
conservan como referencia; la biblioteca nueva tiene versión v6.

```powershell
# Duelo pareado (mismos tiros para todos los porteros).
.venv/Scripts/python.exe tools/evaluate_keeper_fusion.py --keepers scripted vision pose pose_v4 ball_v5 full_v5 --episodes 50 --seed 70001 --output runs/keeper_fusion/final_70001.json --log runs/keeper_fusion/final_70001.log
# Trazas del observador y ajuste del modelo de orientación (desarrollo).
.venv/Scripts/python.exe tools/evaluate_keeper_fusion.py --keepers observer --episodes 25 --seed 69001 --frames --output runs/keeper_fusion/observer_69001.json
.venv/Scripts/python.exe tools/fit_kick_heading.py --frames runs/keeper_fusion/observer_69001.frames.jsonl --report runs/keeper_fusion/heading_fit.json --output models/vision/kick_heading_v1.json
# Tablas y desglose por cruce real.
.venv/Scripts/python.exe tools/analyze_keeper_fusion.py --report runs/keeper_fusion/final_70001.json --observer runs/keeper_fusion/observer_70001.frames.jsonl
# Sólo el duelo (reintenta trabajos si un proceso muere): run_keeper_fusion_step2.bat
```

This product includes software developed by B-Human (http://www.b-human.de).
