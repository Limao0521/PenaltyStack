# Modelo NAO V6 de B-Human

Fuente sin modificar: [`Config/Scenes/Includes/NaoV6H25.rsi3`](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Config/Scenes/Includes/NaoV6H25.rsi3),
revisión oficial B-Human `d89d603f0388208cecd1c3144b6e8eb92aef4a93`.
SHA-256 de `NaoV6H25.rsi3`:
`73f8c9a78b35fbdd18e6e407193924b2d1996812ff78cd405377cc57b0981369`.
El submódulo SimRobot de esa revisión apunta a
[`bhuman/SimRobot@61efef6`](https://github.com/bhuman/SimRobot/tree/61efef6bbe4761889a65c2cd67913af19513e957).

B-Human describe este NAO con 25 cuerpos, 24 articulaciones, masas, inercias,
geometrías de contacto y mallas visuales en el archivo RSI3. Su SimRobotCore3
usa MuJoCo como motor físico. `tools/build_bhuman_model.py` convierte esa
descripción a `nao.xml`, `initial_pose.json` y 50 archivos OBJ usados por
el robot en `meshes/`.
El simulador de penales carga dos instancias de esa conversión; no enlaza ni
ejecuta las bibliotecas C++ originales de SimRobot.

## Alcance de la conversión

- Conserva vértices, triángulos y normales disponibles, marcos/ejes y límites de articulación,
  masas/inercia, cajas/esferas de colisión y parámetros P/D/fuerza del RSI3.
- Usa colores planos derivados de las superficies B-Human; la piel principal
  se aclara para visualizar el NAO blanco. No convierte texturas, cámaras,
  sensores ni la semántica completa del controlador SimRobot.
- El HipYawPitch físico se representa con dos bisagras acopladas y un solo
  actuador, dejando 23 acciones articulares por NAO. Las mallas no colisionan.
- El flujo de penales utiliza raíces libres y los motores originales B-Human
  para actuar sobre los servos. IMU y cargas de pie se calculan desde MuJoCo;
  el balón recibe únicamente contacto físico. El entorno `JointSpaceEnv`
  sigue siendo una interfaz articular sin política para futuros ensayos RL.

La fuente y sus archivos generados están bajo los términos de
[`License.md`](License.md). La licencia permite redistribución con avisos y
contiene condiciones específicas para uso de código B-Human en RoboCup,
incluido el anuncio previo en la lista SPL. La licencia advierte que algunos
elementos de la distribución tienen titulares de derechos distintos. Revisar
esos términos antes de usar el modelo en competición o publicar un derivado.

Regeneración desde el archivo fijado:

```powershell
.\.venv\Scripts\python.exe tools/build_bhuman_model.py
```
