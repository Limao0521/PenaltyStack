# Auditoría PenaltyStack · 2026-10-19

Stack de penales para NAO v6 · **Copa NAO CCM MX 2026, categoría Tiro Penal** · basado en
SabanaHerons2026 (B-Human 2023).

| Dato | Valor |
| :--- | :--- |
| Fecha de auditoría | 2026-10-19 |
| Competencia | 2026-11-03 · Tec de Monterrey CCM |
| Días restantes al auditar | 15 |
| Rama / commit | `despliegue/copa-nao-ccm-2026` @ `17b5c03`, árbol limpio |

## Resumen ejecutivo

El proyecto es un stack completo, disciplinado y bien documentado: simulación MuJoCo con el
motor B-Human original verificado por hash, políticas RL con pruebas ciegas, puerto C++ al
framework del robot con **paridad de 35 686 comprobaciones sin fallos**, cross-compilación
funcionando y despliegue probado en simulacro. Las reglas de la categoría están implementadas
y assertadas en tests. La brecha que bloquea la competencia es una sola: **ningún código se ha
ejecutado jamás en un NAO físico**. Todas las cifras deportivas (93/100 goles, 86/100 atajadas)
son de simulación con percepción idealizada. Con 15 días por delante, el trabajo no es escribir
código nuevo: es validar físicamente lo que ya existe.

Cifras clave al momento de la auditoría:

| Métrica | Valor | Nota |
| :--- | :--- | :--- |
| Ejecuciones en robot físico | **0** | Brecha bloqueante |
| Paridad C++↔Python de políticas | 35 686 / 0 | checks / fallos |
| Suite de tests | 373 / 381 | 8 errores por `psutil` sin declarar (corregido el mismo día, ver §8) |
| Goles `striker_v2` vs FSM base (sim) | 93 / 100 | 12/100 vs portero balanced propio |
| Atajadas `vision_v2_balanced` vs `striker_v2` (sim) | 86 / 100 | 83.75 % en prueba ciega de 160 casos |
| Presupuesto interno de ejecución | 28 s | Regla 3.3.5: 30 s |
| Grafo de módulos del escenario | 136 / 0 | módulos / errores |
| Robots registrados | 9 | diálogo de despliegue, IP 192.168.49.2–.10 |

### Estado por área

| Área | Simulación | Software para hardware | Hardware físico |
| :--- | :--- | :--- | :--- |
| **Visión de balón** | Validada: error mediano 1.19 cm; 74/80 atajadas holdout | Sin calibrar: `CopaOrangeBallPerceptor` YUV, umbrales «a ajustar en la sede» | **Nunca ejecutada** |
| **Visión de cancha** | No existe (estado privilegiado del mundo) | No existe (odometría muerta desde colocación manual) | Deriva sin medir |
| **Decisión tirador** | Validada: 93/100 vs FSM, 12/100 vs nuestro portero | Portada: MLP determinista con paridad verificada | Desplazamiento sim→real sin cuantificar |
| **Decisión portero** | Validada: 83.75 % en prueba ciega de 160 casos | Portada: `LibraryKeeper` balanced/original | **Defensas sin probar en duela** |
| **Actuación (motor)** | Verificada: patadas con contacto, saltos + GetUp, penal completo ×3 | Skills originales: `WalkToBallAndKick`, `DiveSkill` | **Nunca ejecutada en NAO** |
| **Despliegue** | Dry-run cubierto por test | Binario compilado 13.7 MB (2026-10-15), sha256 verificado | **0 despliegues reales** (`runs/robot_logs/` vacío) |
| **Tests** | 381 tests · ~160 s · 8 errores solo por dependencia faltante | — | No aplicable |

## 1 · Contraste con el reglamento (sección 3.3)

Verificado contra `sim/scene.xml`, `robot/Config/`, `tests/test_robot_deploy.py` y el PDF del
reglamento. La geometría de la simulación coincide con la cancha de referencia.

| Regla | Valor | Estado en el proyecto |
| :--- | :--- | :--- |
| Cancha (3.3.4) | 240 × 240 cm | Coincide: líneas a ±1.2 m en la escena |
| Portería | 150 × 80 cm, postes azules | Coincide: postes en y=±0.75, travesaño z=0.8, color azul |
| Área chica (portero confinado) | 60 × 220 cm | Coincide; el portero no camina: defiende en el sitio |
| Marca penal → línea de gol | 130 cm | Coincide: marca en x=0.1, gol en x=−1.2; assertado en `test_competition_safe_defaults` |
| Tirador detrás de la marca | 20 cm | Coincide: spawn del tirador en x=0.3; `strikerBehindMark=200` |
| Balón oficial | naranja Ø65 mm, 55 g | Coincide: r=0.0325 m, masa 0.055 kg en escena y `ballSpecification.cfg` |
| Ejecución tras silbato (3.3.5) | 30 s | Cumple con margen: presupuesto interno 28 s; sin medir con tiempos reales |
| 100 % autónomo, sin Wi-Fi (2.7/3.3.3) | — | Implementado: todo onboard en C++; perfil Wi-Fi `NONE` en el despliegue |
| Desactivación manual, sensor fuera de piernas | — | Implementado: botón trasero de cabeza 0.5 s → ABORT; sin probar en físico |
| Activación ≤10 s tras silbato (2.7) | — | Implementado: botón medio o silbato (`startMode`); por defecto solo botón, anti-silbatos de otras canchas |
| No iniciar antes del silbatazo | — | Implementado: fase `waiting` inmóvil + retardo de armado de 2 s |
| 3 robots, ≥1 tiro cada uno (3.3.3) | — | Implementado, **sin ensayar**: ambos roles viajan en cada robot y se conmutan en cancha (botón frontal 2 s) |
| Superficie: duela de madera | — | **Divergencia**: la simulación usa piso verde tipo césped; los umbrales de color y toda la validación visual se hicieron sobre verde |
| Detalles visuales menores | marca Ø10 cm, líneas 1.9 cm | Cosmético: en sim marca Ø3.2 cm, líneas 1.2 cm; nadie las detecta hoy, pero importan si se añade corrección visual |

## 2 · Visión y detección

### Balón — dos stacks paralelos

- **Python (sim)** `penalty/perception.py`: umbral RGB + flood-fill, proyección al plano del
  suelo con intrínsecos calibrados, `BallTracker` con gates de edad/outliers. Validado: error
  mediano de posición 1.19 cm (p95 1.52), velocidad 0.016 m/s; portero rinde 74/80 en holdout.
  Todo sobre renders de MuJoCo con piso verde.
- **C++ (robot)** `CopaOrangeBallPerceptor`: segmenta naranja en YUYV, máscara de cuerpo
  propio, filtro por tamaño proyectado del balón de 65 mm. **Nunca ejecutado con imágenes
  reales** — su propio `.cfg` dice «thresholds to be tuned with images from the competition
  venue». Hay riesgo documentado de que la madera clara se acerque al naranja con mucha luz.
- **Paridad**: la lógica posterior al píxel (`BallTracker`, proyección de decisión, keeper)
  está portada a `CopaPolicies.h` y cubierta por la paridad 35 686/0. El detector de píxel en
  sí no tiene paridad: RGB/8-conexo en Python vs YUV/4-conexo en C++; son conceptualmente
  análogos, numéricamente independientes.

### Cancha, líneas y portería — no existe detección

- No hay detector de líneas, frontera de campo ni portería en ningún lado. La localización es
  **colocación manual por el capitán + odometría muerta** desde la pose inicial configurada
  (`CopaPenaltyTactics::fieldPose()`).
- El `FieldBoundaryProvider` de B-Human arranca pero no se consume; la documentación declara la
  autolocalización por líneas «no fiable» sobre duela con líneas negras.
- **Fallo silencioso**: si el robot es movido, se cae o deriva, el marco de campo se corrompe
  sin aviso — el portero asume su línea a −0.12 m fijos y el tirador calcula bearings a postes
  desde odometría.

### Rival / tirador — stack experimental completo, fuera de la ruta recomendada

- Detector de robots B-Human original (ONNX + rama de contraste) y yaw por 2 294 plantillas
  (`nao_visual_yaw_v1`, error mediano 1.1–1.7° en sim). Experimental.
- La red ONNX superior tuvo **0 detecciones en 30 renders de MuJoCo** (gap de dominio); solo la
  rama inferior de contraste funcionó (28/30, error 11.4 cm). El host nativo nunca se compiló
  para Linux/NAO.
- El portero recomendado (`vision_v2_balanced`) decide **solo por balón**: la fusión con
  postura/orientación (v3/v4/v5) no aportó mejora medible (+0.2 pp, 1 decisión cambiada de 400)
  y quedó descartada con evidencia.
- En el robot, el tirador **no percibe al portero**: usa su pose nominal reglamentaria en la
  línea de gol (permitido por 3.3.5, y coherente con la regla de colocación).

### Sensores propios

- `penalty/sim_sensors.py` aproxima bien el hardware (encoders, gyro integrado, FSR desde
  contactos, odometría por pie de apoyo), pero es **ideal**: sin ruido, latencia ni
  cuantización; el motor nativo conserva actitud propia ideal (`mujoco-truth-v2`). La deriva de
  odometría real sobre duela no está modelada.
- Los perfiles NAOqi tienen calibraciones de cámara placeholder (`fx:0, "TO_FILL"`) que el
  empaquetador rechaza: la ruta Python 2 queda como solo-observación por diseño.

## 3 · Toma de decisiones y actuación

### Arquitectura

Tres capas limpias: **L3 táctica** (RL/reglas) → **L2 FSM + skills parametrizadas**
(`penalty/control.py`, puertas de seguridad, commit único) → **L1 motor B-Human original
compilado** (559 archivos preservados por hash; WalkingEngine, KickEngine,
KeyframeMotionEngine, FallEngine). Los contratos (`rl_obs_v1` 19 campos, `vision_state_v1/v2`,
`keeper_action_v2`) se respetan con gobernanza explícita: nunca se edita un contrato publicado,
se versiona. Los mismos contratos los consumen Python (sim) y C++ (robot), que es lo que hace
creíble la paridad.

### Políticas recomendadas (consistente en docs, manifests y registro del robot)

- **Tirador `striker_v2`** (recomendado) — PPO sobre `rl_obs_v1`: 93/100 goles vs FSM base,
  12/100 vs nuestro portero balanced; reparte 51/49 % y apunta a esquinas |y|≈0.64 m; potencia
  → alcance pedido 1500–2900 mm.
- **Portero `vision_v2_balanced`** (recomendado) — trayectoria de balón como señal primaria:
  confirmación de 2 capturas, mediana de ≤3 proyecciones de cruce, banda central ±0.30 m,
  saltos solo con llegada ≤1.8 s, commit único e irrevocable. 86/100 atajadas vs striker_v2;
  83.75 % en prueba ciega.
- **Todo lo demás es experimental sin promoción**: v3 visual, v4 preserving, selector de 6
  defensas (84.375 %), perfil de movimiento (85 %), fusión v5 — las ganancias no superan los
  intervalos de confianza frente a los recomendados. Disciplina de evaluación correcta: tests
  ciegos fijados antes de entrenar, semillas disjuntas, IC pareados.
- Contexto competitivo honesto: el techo del portero hoy es **el movimiento, no la decisión**
  (la estirada de B-Human alcanza 0.46–0.59 en sim; la guardia central tras el disparo llega a
  0.55 y ningún PPO la superó).

### Actuación — verificada en sim solo para el caso nominal

- Verificado físicamente en sim: stand y marcha sin caída; ambas patadas con contacto
  pie-balón exigido y desplazamiento ~1.0 m a 2500 mm pedido; saltos laterales con GetUp
  automático y estabilidad; penal completo ×3 con resultado y determinismo bit a bit.
- **No verificado**: otros alcances de patada («alcance pedido ≠ distancia garantizada»; borde
  de caída documentado en 2242/2248 mm); solo el parámetro `power` llega al runtime nativo;
  latencias reales (en sim la inferencia no hace avanzar el reloj físico); superficies reales.
- En el robot: `WalkToBallAndKick` (alineación precisa, sin turn-kick) y `DiveSkill` con 6
  defensas; `allowJumpDives=false` degrada los saltos a squats anchos sin caída — buen
  mitigador para la duela. Patada liberada una sola vez con balón ≤400 mm, timeout 8 s, viaje
  máximo del tirador 900 mm (nunca persigue).

### Tiempos

FSM a 100 ms y motor a 12 ms tanto en sim como en el robot. Presupuesto 28 s vs 30 s
reglamentarios: margen correcto, **sin verificar con tiempos reales** — no existe medición
end-to-end cámara→decisión→motor en el NAO. El objetivo interno documentado es un penal
completo <15 s.

## 4 · Despliegue y pruebas

### Ruta a hardware

- `robot/`: framework B-Human 2023 completo (SabanaHerons2026 @`2360ab1`, 4 994 archivos con
  procedencia SHA-256). 15 originales modificados (mínimos y acotados), **26 archivos nuevos**:
  `CopaPenaltyController` (FSM de fases y gestos de botones/silbato), `CopaPenaltyTactics`
  (odometría, tracking, políticas), `CopaPolicies.h` (747 líneas, solo stdlib: MLP
  determinista, BallTracker, VisionKeeper/LibraryKeeper), `HandleCopaNAOPenalty` (skills
  originales), `CopaOrangeBallPerceptor`.
- Verificado en vivo durante la auditoría: cross-compilación WSL/clang 18 (7/7 herramientas,
  binario Release de 13.7 MB del 15-oct), paridad C++↔Python 35 686/0, grafo de módulos
  136/0, dry-run real del script de despliegue.
- `DeployCopaNAO.bat` → diálogo Tk: 9 robots con IP fija (192.168.49.2–.10), rol, políticas por
  registro versionado (`policies.json` con sha256), Wi-Fi off, compilación, despliegue paralelo
  con verificación sha256 en el robot, arranque de `bhuman.service` y descarga de logs. El
  script de despliegue trabaja sobre una copia staging con hard-links y nunca modifica el árbol
  compartido.
- El adaptador NAOqi Python 2 anterior queda como solo-observación por diseño (rechaza
  `kick_enabled`). **Ojo**: `docs/PREPARACION_NAO.md` describe esa ruta legacy y puede
  confundir; la ruta real es `robot/` + `docs/GUIA_DESPLIEGUE_ROBOT_FISICO.md` (banner añadido
  el mismo día de la auditoría, ver §8).

### Pruebas

- **381 tests en 37 archivos, ~160 s** (corrida en vivo de esta auditoría): 373 pasan, 8 errores
  — todos `ModuleNotFoundError: psutil` en el teardown del reentrenamiento RL. `psutil` no está
  en `.venv` ni declarado en `requirements-rl.txt`. Una instalación fresca según la
  documentación reproduce los 8 errores. Arreglo trivial: instalar y declarar (hecho, ver §8).
- Cobertura buena de sim, RL, percepción Python, paridad y despliegue (incluye dry-run en WSL y
  defaults seguros de competencia). **Sin cobertura**: la lógica C++ de los módulos nuevos (FSM
  de botones, segmentador naranja) más allá del núcleo portable; nada de hardware (imposible
  por ahora).
- `runs/robot_logs/` no existe → confirma que ningún robot ha corrido el escenario.

## 5 · Brechas priorizadas para el despliegue real

### P0 · Bloqueantes

1. **Cero ejecución en hardware.** Toda la cadena percepción→decisión→actuación existe solo en
   simulación y simulacro. Gate máximo: un penal completo autónomo (botón/silbato, sin PC ni
   Wi-Fi) en cada uno de los 3 robots. Requiere tener los robots a mano ya: los robots Sabana
   Herons ya tienen la base B-Human; un NAO de fábrica necesitaría la instalación completa de
   `robot/Install/` antes de todo lo demás.
2. **Detección del balón naranja sin calibrar en la sede.** Umbral YUV calculado contra madera
   y piel pero nunca probado con la duela y la iluminación reales. Es el único punto del que
   dependen ambos roles. Capturar imágenes con las dos cámaras (balón a 0.2/1.3/2 m) y ajustar
   `copaOrangeBallPerceptor.cfg` sin recompilar; vigilar madera clara con mucha luz (subir
   `minChroma`) y reflejos (`maxY`). → Toolkit de calibración listo (§8 y
   `docs/CALIBRACION_BALON.md`); falta ejecutarlo con el robot físico.
3. **Calibración de cámara y cinemática por robot.** La proyección del balón al suelo (y por
   tanto la patada y la defensa) depende de intrínsecos/extrínsecos heredados de SabanaHerons.
   Verificar en cada unidad antes de confiar en `WalkToBallAndKick`.
4. **Actuación física nunca probada.** Secuencia incremental obligatoria: patada única desde la
   marca → aproximación → patada con política (medir alcance real vs pedido y tiempos) →
   portero con `--jumps off` → GetUp tras caída real → estiradas con caída solo si la
   superficie y el hardware lo toleran.

### P1 · Robustez

5. **Deriva de odometría sin percepción de cancha.** No hay corrección visual: medir la deriva
   real caminando sobre duela (p. ej. recorrido de 1 m ida/vuelta y comparar). El portero no
   camina (riesgo bajo); el tirador orbita hasta 900 mm (riesgo medio). Si la deriva es grande,
   reducir el viaje del tirador o añadir una verificación de posición previa al tiro.
6. **Desplazamiento sim→real del tirador.** Entrenado con observación privilegiada ruidosa;
   desplegado con BallPercept + odometría + portero nominal. Cuantificarlo en sim alimentando
   la política con el camino de observación del C++ (cámaras alternadas con gracia de 100 ms,
   frames retrasados, ruido del tracker) antes de viajar.
7. **Latencias reales desconocidas.** Medir onboard el ciclo cámara→decisión→motor y el coste
   de `WalkToBallAndKick`/`Dive`. Verificar que el penal completo cabe en 30 s con margen
   (objetivo interno <15 s).
8. **Protocolo competitivo sin ensayar.** Rotación de los 3 robots (cada uno ≥1 tiro), cambio
   de rol en cancha, rearme tras tiro repetido (botón frontal), desactivación de emergencia
   (botón trasero), arranque ≤10 s tras silbato. Todo implementado; nada ensayado en físico.
   Ojo operativo: el capitán debe colocar al portero exactamente como asume la configuración
   (raíz ~12 cm delante de la línea), o la geometría se corrompe en silencio.

### P2 · Higiene

9. **Entorno y documentación.** Añadir `psutil` a requirements y dejar la suite verde;
   corregir el conteo vivo de archivos nuevos del robot (25→26); aclarar en
   `PREPARACION_NAO.md` que describe la ruta legacy; opcional: acercar la apariencia de la sim
   a la sede (piso duela, marca Ø10 cm) si se quiere re-validar el detector en sim. Las
   menciones de «367 pruebas» en `docs/RESUMEN_PRUEBAS.md` y `docs/RESULTADOS_RL.md` son
   evidencia histórica del reentrenamiento `striker_preserving_v4` (su log archivado registra
   367) y se conservan a propósito. → Todo lo no opcional quedó aplicado (§8).

## 6 · Plan de trabajo · 2026-10-20 → 2026-11-03

Supone acceso inmediato a al menos un NAO v6 con la base B-Human y a una superficie de duela
similar a la sede. Si el acceso a robots se retrasa, todo el plan se desplaza: esa es la
dependencia crítica.

| Fechas | Objetivo | Actividades | Criterio de salida |
| :--- | :--- | :--- | :--- |
| Lun 20 oct | **Arranque en físico** | Entorno ya corregido (§8): suite 412/412. Despliegue real en robot #1 (ping → dry-run → deploy con `DeployCopaNAO.bat`). Arranque vigilado con robot sujeto: protocolo completo de botones (pecho=de pie, medio=start, trasero=ABORT, frontal=rearme/cambio de rol, 3 botones=unstiff) sin balón. | `bhuman.service` activo; logs `[CopaPenalty]` correctos; gestos responden; Wi-Fi NONE confirmado. |
| Mar–Mié 21–22 | **Visión calibrada** | En duela con la luz más parecida a la sede: capturar balón naranja a 0.2/1.3/2 m con ambas cámaras (ver `docs/CALIBRACION_BALON.md`). Ajustar `copaOrangeBallPerceptor.cfg` sin recompilar. Verificar tracking estable en logs. Medir deriva de odometría en marcha real (recorrido conocido). | Detección estable >95 % de frames a las 3 distancias; sin falsos positivos con la cancha vacía; deriva cuantificada y aceptable. |
| Jue–Vie 23–24 | **Tirador incremental** | Patada única desde la marca (2500 mm) midiendo distancia real; luego aproximación completa (spawn a 20 cm detrás de la marca); luego política `striker_v2` end-to-end. Verificar que el viaje no excede 900 mm y el timeout de 8 s no dispara. | ≥8 de 10 patadas con contacto y alcance útil (>1.3 m llega a portería); tiempos medidos dentro del presupuesto de 28 s. |
| Sáb–Dom 25–26 | **Portero incremental** | Guardia central y squats anchos con `--jumps off`; GetUp tras caídas reales; si el hardware y la superficie responden, activar `allowJumpDives` progresivamente. Primer penal completo tirador-vs-portero en físico. | Defensas ejecutan y el robot se recupera solo; decisión de saltos on/off tomada con evidencia, no por optimismo. |
| Lun–Mié 27–29 | **Ensayo de competencia** | Desplegar los 3 robots. Simular series de 5 tiros con rotación (cada robot tira ≥1 vez), cambio de rol en cancha, tiros repetidos (rearme), activación ≤10 s. Todo sin PC ni Wi-Fi, cronometrando. Descargar `runs/robot_logs` como evidencia. Ajustar fino de políticas/defensas por defecto. | Serie completa de 5 tiros ejecutada por los 3 robots sin intervención; penal medio <15 s; checklist de `docs/robot/03` §7 en verde. |
| Jue–Vie 30–31 | **Congelación y contingencia** | Congelar código. Re-correr `tests.test_robot_policy_parity` + `tests.test_robot_deploy` + validación de escenario. Preparar presets de respaldo (portería `center_guard` si la visión falla; `--jumps off` por defecto si hay duda). Kit de viaje: baterías, Ethernet, binario y Config versionados, checklist impreso. | Suite verde sobre el código congelado; manifiestos de despliegue con sha256 guardados; plan B documentado en una página. |
| Sáb–Dom 1–2 nov | **Ensayo ligero** | Solo repetición de protocolo (arranque, activación, ABORT, rearme). Nada nuevo. Descanso del hardware. | Los 3 robots arrancan y se detienen de forma fiable con el gesto de competencia. |
| Lun 3 nov | **Competencia** | Zona de calibración: máx. 5 min por prueba — llevar presets, no sesiones de ajuste. Verificar luz de la cancha oficial contra la calibrada; si difiere, mover `minChroma`/`maxY` según lo ensayado. Ethernet solo antes del bloque; desconectado en el área. | Serie de 5 tiros con los 3 robots tirando; sin descalificaciones por protocolo. |

> **Regla de oro del plan**: cada paso físico es incremental y reversible (sujeto → sin balón →
> balón estático → patada → penal completo). La documentación del proyecto ya lo dice: «no
> habilitar movimiento físico sin calibración y ensayos incrementales». No saltarse etapas por
> calendario.

## 7 · Riesgos competitivos (rankeados)

| # | Riesgo | Por qué | Mitigación |
| :--- | :--- | :--- | :--- |
| 1 | **Detección del balón bajo la luz de la sede** | Umbral YUV nunca probado en real; madera clara + iluminación desconocida; ambos roles dependen de esto. | Calibrar días 21–22 en superficie similar; presets guardados; portería `center_guard` como plan B que minimiza dependencia del tracking fino. |
| 2 | **Desconocidos de la primera ejecución** | Servicio, cámaras alternadas (gracia 100 ms sin medir), latencias reales, arranque en frío: nada se ha ejecutado en físico. | Arranque vigilado el día 20; logs `[CopaPenalty]` como primera fuente de verdad; margen 28 s vs 30 s. |
| 3 | **Estiradas del portero sobre duela** | Caídas laterales repetidas dañan hardware; alcance real (0.46–0.59 en sim) no transferido. | Empezar con `--jumps off` (squats anchos sin caída); activar saltos solo con evidencia; la regla balanced ya gana sin saltos ajustados. |
| 4 | **Desplazamiento sim→real del tirador** | Entrenado con escena privilegiada; desplegado con percepción estimada y portero nominal. | Evaluar en sim con el camino de observación del C++ antes de viajar; en físico, medir alcance real y ajustar `kickLength` si deriva sistemáticamente. |
| 5 | **Deriva de odometría sin percepción de campo** | Colocación manual + odometría muerta; sin corrección visual; fallo silencioso. | Medir deriva real; colocar con plantilla física (marca en el suelo); protocolo de re-colocación entre tiros; el portero no camina y el tirador viaja ≤900 mm. |

## 8 · Correcciones aplicadas durante la auditoría (2026-10-19)

### Durante la auditoría

1. **`psutil` declarado e instalado** (cierra el único P2 de tests). Se añadió
   `psutil>=5.9,<8` a `requirements-rl.txt` y se instaló en `.venv`; la reejecución de los
   módulos que fallaban (`penalty/rl/striker_retraining*.py`) pasa 28/28. Los 8 errores de la
   suite completa eran exclusivamente esta dependencia faltante.
2. **Recuento de archivos nuevos corregido**: `docs/robot/02_PLAN_E_IMPLEMENTACION.md` decía
   «25 archivos nuevos»; son **26**, verificado con `tools/robot_build.py --check-originals`
   (`changed=15, missing=0, added=26`).
3. **Divergencia de rutas de despliegue resuelta en la documentación**:
   `docs/PREPARACION_NAO.md` llevaba a pensar que el adaptador NAOqi Python 2 era la ruta de
   despliegue. Ahora un banner al inicio aclara que esa ruta es **legacy, solo de
   observación**, y que la ruta correcta para la competencia es el framework B-Human de
   `robot/` con `DeployCopaNAO.bat` (→ `docs/GUIA_DESPLIEGUE_ROBOT_FISICO.md`). Queda
   conservada como contexto histórico, igual que `docs/DESPLIEGUE.md`.
4. **Las menciones de «367 pruebas» se conservan deliberadamente**: son evidencia histórica
   ligada a logs archivados (`docs/validation/rl/striker_preserving_v4/full_tests.log`); la
   cifra actual de la suite se reporta en cada corrida, no se reescriben los registros.

### Paquete de calibración del balón (seguimiento inmediato, sin hardware)

Cierra la brecha de herramientas del P0-2 («calibrar balón en sede»): el flujo completo
capturar → ajustar → desplegar Config → verificar en vivo queda listo para ejecutarse en
cuanto haya robot físico. Ningún archivo del framework fue modificado.

5. **Tres herramientas nuevas**:
   - `tools/nao_capture_ball.py` — corre en el robot (Python 2.7/NAOqi): captura frames
     YUV422 nativos (o RGB como respaldo) de ambas cámaras por lotes etiquetados, con
     sondeo de colorspaces y validación del tamaño de buffer (las constantes cambian entre
     versiones de NAOqi; nada se asume).
   - `tools/capture_ball_calibration.py` — driver en el PC (vía WSL, como el despliegue):
     sube el script, detiene/reinicia `bhuman.service` (B-Human sostiene la cámara por
     V4L2), captura y trae los frames a `runs/ball_calibration/<robot>/<lote>/`.
   - `tools/tune_ball_thresholds.py` — espejo Python de la segmentación de
     `CopaOrangeBallPerceptor` (clasificación YUV, rejilla paso 4, flood fill 4-conexo,
     filtros de forma): evalúa los umbrales actuales contra capturas reales, busca los
     umbrales factibles de **mayor margen bilateral** cambiando lo mínimo, y con `--apply`
     reescribe solo las 5 claves de color del `.cfg` (respaldo `.bak`), con `--report` JSON
     como evidencia.
6. **31 tests nuevos**: `tests/test_tune_ball_thresholds.py` (17: conversión de color,
   espejo de segmentación, márgenes, sugerencia, carga de capturas, fin-a-fin con `.cfg`
   temporal) y `tests/test_capture_ball_calibration.py` (14: comandos ssh/rsync,
   validaciones anti-inyección de shell, higiene de imports —el driver no puede insertar
   `deploy/` en `sys.path` sin sombrear el paquete `deploy`—, forma del script py2).
   Suite completa: **412 tests en verde**.
7. **Documentación**: `docs/CALIBRACION_BALON.md` — protocolo de capturas por lote
   (`ball_020/130/200`, `bg_cancha`, `bg_distractores`), ajuste, despliegue solo-Config y
   criterios de aceptación en vivo; enlazada desde `GUIA_DESPLIEGUE_ROBOT_FISICO.md` §8 y
   §10.
8. **Validación del ajustador sin hardware** (imágenes sintéticas): un distractor «casi
   naranja» (YUV 150,100,185) es falso positivo con los umbrales actuales — reproduce el
   riesgo #1 de esta auditoría — y el ajustador encuentra la corrección mínima
   (`minChroma` 80→120, márgenes fondo 35 / balón 41), coherente con la regla manual «sube
   `minChroma`» de la guía física.

### Lo que sigue pendiente con hardware

Los P0-1 (primer arranque vigilado), P0-3 (tiro físico) y la ejecución real del P0-2
requieren el robot; este paquete elimina el bloqueo de herramientas, no la validación
física. Ver el plan de trabajo §6.

## Método de la auditoría

Tres auditorías de área independientes (visión/detección, decisión/actuación,
despliegue/tests) más verificaciones en vivo: suite completa de tests (381 en 161.8 s: 373
pasan, 8 errores por `psutil`), reejecución del módulo fallido, git (rama limpia @ `17b5c03`),
contraste del reglamento (PDF oficial) contra `sim/scene.xml` y configs del robot, y
verificación de cross-compilación, paridad (35 686/0) y grafo de módulos (136/0). Las métricas
RL citadas provienen de las pruebas ciegas documentadas en `docs/RESUMEN_PRUEBAS.md` y
`docs/validation/`; no se reentrenó ni reejecutó nada de RL. Fuentes principales: `README.md`,
`.ai/HANDOFF.md`, `docs/robot/01-03`, `docs/GUIA_DESPLIEGUE_ROBOT_FISICO.md`,
`docs/VISION_PORTERO.md`, `docs/PORTERO_FUSION.md`, `docs/RESULTADOS_RL.md`, `AGENTS.md`.
