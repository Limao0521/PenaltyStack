# Calibración del balón naranja en la sede

El perceptor del robot (`CopaOrangeBallPerceptor`) decide qué píxeles son «naranja» con cinco
umbrales YUV que su propio `.cfg` manda ajustar con imágenes de la sede: la duela de madera,
la iluminación del Tec de Monterrey CCM y las cámaras de cada NAO son desconocidas hasta
estar ahí. De esa detección dependen **ambos roles** (el tirador rastrea el balón para
caminar y patear; el portero decide la estirada por su trayectoria). El reglamento da como
máximo **5 minutos por prueba en la zona de calibración**: el objetivo de este flujo es llegar
con presets validados, no a improvisar una sesión en cancha.

Herramientas (todas nuevas, ninguna toca código del framework):

| Pieza | Dónde corre | Función |
| :--- | :--- | :--- |
| `tools/nao_capture_ball.py` | En el robot (Python 2.7/NAOqi) | Captura frames YUV422 nativos (o RGB) de ambas cámaras con una etiqueta |
| `tools/capture_ball_calibration.py` | En el PC (WSL, como el despliegue) | Sube el script, detiene/reinicia `bhuman`, captura y trae los frames a `runs/ball_calibration/` |
| `tools/tune_ball_thresholds.py` | En el PC (`.venv`) | Replica la segmentación C++ sobre las capturas, evalúa los umbrales actuales, sugiere los de mayor margen y los aplica al `.cfg` |
| `robot/Config/Scenarios/CopaNAOPenalty/copaOrangeBallPerceptor.cfg` | Se despliega sin recompilar | Los cinco umbrales (`minY/maxY/maxU/minV/minChroma`) y filtros de forma |

Pruebas: `tests/test_tune_ball_thresholds.py`, `tests/test_capture_ball_calibration.py`.

## 0. Flujo completo

```
sede: capturar ──► PC: ajustar ──► desplegar Config ──► robot: verificar
(ball_*/bg_*)      (sin recompilar)   (logs [CopaPenalty])
```

Requisitos previos: robot desplegado según la
[guía del robot físico](GUIA_DESPLIEGUE_ROBOT_FISICO.md) §3–§5, cable Ethernet (regla 2.7:
sin Wi-Fi), balón oficial naranja de 65 mm y —idealmente— una superficie de duela con luz
parecida a la del bloque de competencia.

## 1. Capturar en la sede

Con el robot encendido y cableado, desde el PC (el driver detiene `bhuman` porque B-Human
sostiene la cámara por V4L2, y lo reinicia al terminar):

```powershell
python tools\capture_ball_calibration.py --capture falcanao --label ball_020 --count 20
python tools\capture_ball_calibration.py --capture falcanao --label ball_130 --count 20
python tools\capture_ball_calibration.py --capture falcanao --label ball_200 --count 20
python tools\capture_ball_calibration.py --capture falcanao --label bg_cancha --count 20
python tools\capture_ball_calibration.py --capture falcanao --label bg_distractores --count 20
```

Protocolo por lote (ambas cámaras por omisión: `--cameras 0,1`; VGA 640×480 como el runtime):

| Lote | Qué encuadrar | Por qué |
| :--- | :--- | :--- |
| `ball_020` | Balón a 20 cm de la cámara, centrado y también a ±30° hacia los bordes | Distancia mínima del rastreo del tirador; la etiqueta codifica la distancia para el chequeo de radio |
| `ball_130` | Robot en pose de tiro, balón en la marca penal (1,3 m) | Geometría real del disparo |
| `ball_200` | Balón a 2 m, a nivel del suelo | Alcance del portero |
| `bg_cancha` | Cancha **vacía**: duela, líneas negras, portería azul, sombras | Cero falsos positivos exigidos |
| `bg_distractores` | Lo que pueda confundir: duela clara con luz directa, piel (manos/pies de gente en cuadro), ropa naranja de espectadores, postes | La madera clara y la piel son el riesgo documentado en el `.cfg` |

Reglas de captura:

- **Etiquetas**: el clasificador usa el prefijo — `ball*` exige detección, `bg*` exige cero
  blobs. `ball_<cm>` activa además el chequeo de radio esperado por pinhole.
- **Repetir por robot** (las cámaras difieren entre unidades): al menos en los 3 que tiran
  (regla 3.3.3). Guardar todo bajo `runs/ball_calibration/<robot>/`.
- Con la luz más parecida posible a la del bloque de competencia; si la sede tiene luz mixta
  (sol + lámparas), capturar en las dos condiciones.
- El robot captura con exposición y balance de blancos automáticos, igual que el runtime
  (`Config/Locations/Default/cameraSettings.cfg`); por eso estas capturas aproximan bien el
  feed real. La validación final (§3) es siempre contra el perceptor vivo.

Sin el driver (manual): detener el servicio y correr el script en el robot —

```bash
systemctl --user stop bhuman.service
python /home/nao/ball_calibration/nao_capture_ball.py --label ball_130 --cameras 0,1 --count 20
systemctl --user start bhuman.service
```

(el script se sube con `python tools\capture_ball_calibration.py --push falcanao`).

Traer las capturas al PC:

```powershell
python tools\capture_ball_calibration.py --fetch falcanao
```

Quedan en `runs/ball_calibration/falcanao/<lote>/` con su `capture_meta.json` (dimensiones,
formato, cámara). `runs/` no se versiona.

## 2. Ajustar los umbrales en el PC

```powershell
.\.venv\Scripts\python.exe tools\tune_ball_thresholds.py --captures runs\ball_calibration\falcanao --report runs\ball_calibration\falcanao\report.json
.\.venv\Scripts\python.exe tools\tune_ball_thresholds.py --captures runs\ball_calibration\falcanao --apply
```

Qué hace el ajustador:

- **Replica la segmentación C++** celda por celda: clasificación YUV
  (`Y∈[minY,maxY] ∧ U≤maxU ∧ V≥minV ∧ V−U≥minChroma`), rejilla de paso 4 muestreada en
  `gx*step+step/2`, flood fill 4-conexo y filtros de forma (`minBlobCells=3`,
  `minFillRatio=0.45`, `maxAspectRatio=2.0`). No replica (son del runtime): el recorte
  BodyContour — se aproxima con `--clip-bottom PX` —, la proyección al suelo ni las puertas
  `maxDistance`/radio esperado; el chequeo de radio aparece en el reporte como aviso.
- **Evalúa los umbrales actuales** sobre cada captura: `DETECTA`/`NO_DETECTA` (balón),
  `SIN_BLOBS`/`FALSO_POSITIVO` (fondo).
- **Busca la sugerencia**: recorre `maxU`, `minV`, `minChroma` (añade `minY`/`maxY` con
  `--search-y`) y se queda con los umbrales factibles —cero falsos positivos en **todos** los
  fondos, detección en **todas** las de balón— que maximizan el **margen bilateral**:
  cuántos niveles YUV aguanta la elección en ambas direcciones (relajar hasta que el fondo
  clasifique, endurecer hasta perder el balón). A igual margen, cambia lo mínimo respecto al
  `.cfg` actual. `--quick` hace una búsqueda más gruesa.
- **`--apply`** reescribe solo las cinco claves de color del `.cfg` (respaldo `.bak` al lado,
  comentarios intactos) y recuerda el siguiente paso.

Lectura del margen: ≥15 niveles es holgado; <5 significa que las capturas son pobres o la
luz es extrema — capturar más fondos antes de confiar. El `report.json` (umbrales, márgenes,
veredicto por imagen) es la evidencia de la sesión; archivarlo junto con las capturas.

## 3. Desplegar y verificar en el robot

1. Desplegar **solo configuración**: `DeployCopaNAO.bat` → desmarcar **Compilar antes de
   desplegar** → *Desplegar seleccionados* (ver [guía](GUIA_DESPLIEGUE_ROBOT_FISICO.md) §8).
   El `.cfg` viaja con el resto de Config; el binario no se toca.
2. Con `bhuman` corriendo, colocar el balón en la marca penal y observar al tirador
   rastrearlo; luego probar a 0,2 m y 2 m.
3. Descargar y leer los logs:

   ```powershell
   python deploy\robot_deploy.py --logs falcanao
   ```

   Las líneas `[CopaPenalty]` muestran detecciones y decisiones.

Criterios de aceptación (los del plan día 21–22 de la [auditoría](AUDITORIA_2026-10-19.md)):

- Detección estable en >95 % de los frames con el balón a 0,2 / 1,3 / 2 m, ambas cámaras.
- **Cero** falsos positivos con la cancha vacía durante 60 s (duela, líneas, postes, sombras).
- `positionOnField` del balón coherente con la marca (error < 15 cm a 1,3 m).
- Un tiro completo de prueba siguiendo la [guía](GUIA_DESPLIEGUE_ROBOT_FISICO.md) §7.

Si falla: repetir §1–§2 con los casos que fallaron como capturas nuevas (el lote
`bg_distractores` manda).

## 4. Día de competencia (regla de los 5 minutos)

- Llevar por robot: `.cfg` ajustado, `report.json` y un preset de respaldo
  (los umbrales de fábrica del repo sirven como punto neutro).
- En la zona de calibración **no** se ejecuta el flujo completo: se verifica con un balón en
  la marca y, si la luz difiere de la calibrada, se aplican las reglas manuales de la
  [guía](GUIA_DESPLIEGUE_ROBOT_FISICO.md) §8 — no ve el balón: bajar `minV`/`minChroma` o
  subir `maxU` de a 10; confunde madera/piel: subir `minChroma`; reflejos: bajar `maxY`.
- Prioridad operativa: el tirador necesita la detección para caminar y patear; si la visión
  es dudosa, el portero puede jugar el plan B (`center_guard`, que minimiza la dependencia
  del rastreo fino) antes que arriesgar un tirador ciego.

## 5. Detalles técnicos y límites

- **Formato de captura**: YUV422 nativo intercalado (`.yuv`, 2 bytes/píxel) siempre que la
  cámara lo entregue — es exactamente lo que consume el perceptor. Si NAOqi solo da RGB, se
  guarda `.ppm` y el ajustador convierte con BT.601 (±1 nivel frente al YUV de cámara;
  irrelevante contra márgenes de decenas). El script de robot **prueba** los IDs de
  colorspace en orden (12, 108, 13, 11) y valida el tamaño del buffer contra
  `width*height*bpp` en vez de confiar en constantes que cambian entre versiones de NAOqi.
- **Sin `psutil`/`scipy`/`cv2`**: el ajustador solo usa numpy (+PIL para `.ppm/.png/.jpg`),
  que ya están en el entorno.
- **El detector de píxel no tiene paridad C++↔Python automatizada** (a diferencia del núcleo
  de políticas): el espejo del ajustador está probado con imágenes sintéticas contra la
  lógica documentada del `.cpp`, pero la autoridad final es el perceptor vivo (§3). Si se
  cambia la segmentación en C++, hay que actualizar `tools/tune_ball_thresholds.py`.
- **Umbrales por robot**: si las cámaras difieren notablemente entre unidades, cada robot
  lleva su `.cfg` — el despliegue por robot del diálogo (`DeployCopaNAO.bat`) lo permite
  desplegando de a uno con su Config ajustada.
