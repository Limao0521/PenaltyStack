# Baseline de movimiento B-Human

`bhuman/` conserva sin editar el módulo completo
`Src/Modules/MotionControl/` de
[`SabanaHerons2026`](https://github.com/juanulloa2050/SabanaHerons2026/tree/2360ab1100ad28f70e190c893f6b095651ff0bbc),
revisión `2360ab1100ad28f70e190c893f6b095651ff0bbc`: 54 archivos de
MotionControl. Se incluyen además módulos de sensores y `NaoProvider`,
`Tools/Motion`, representaciones de movimiento, configuración motriz y el
cierre recursivo de cabeceras locales: 266 archivos en total. Los SHA-256 y
rutas originales están en `motion_manifest.json`. La licencia del origen se
conserva en `../heritage/upstream/License.md`.

La carpeta es **referencia de código**, no un binario ejecutable. El motor
depende del framework y la cadena de compilación de B-Human que no se
migraron. No se conecta hoy ni al simulador MuJoCo ni al proceso NAOqi.

## Comparación futura con RL

1. Construir y ejecutar el motor de referencia en un entorno B-Human
   compatible, con el mismo NAO V6 y las mismas condiciones de penal.
2. Registrar comandos articulares, poses, velocidades, contacto, trayectoria
   del balón y eventos de caída con tiempo uniforme y nombres de junta.
3. Aplicar a MuJoCo la misma pose inicial, balón, escenario y presupuesto
   de tiempo. `penalty/joint_env.py` expone las 23 acciones controlables por
   NAO del modelo derivado y sus límites; manos y dedos no participan.
4. Medir éxito de gol/bloqueo, caídas, tiempo, desviación del balón y esfuerzo
   de control en ensayos repetidos. Registrar por separado las diferencias
   entre modelo y hardware; no interpretar una ventaja simulada como ventaja
   física sin calibración y pruebas reales.

Para regenerar la copia desde el checkout local fijado:

```powershell
python tools/import_motion_engine.py --source C:\ruta\SabanaHerons2026 --commit 2360ab1100ad28f70e190c893f6b095651ff0bbc
```
