# robot/ — framework del NAO V6 (B-Human 2023 / SabanaHerons2026)

Subconjunto de [SabanaHerons2026](https://github.com/juanulloa2050/SabanaHerons2026) `2360ab1`
(basado en el B-Human Code Release 2023) necesario para compilar y desplegar `bhuman` en
robots NAO V6 reales, con el escenario **CopaNAOPenalty** (Copa NAO CCM MX 2026, Tiro Penal).

- Importado por `tools/import_robot_framework.py`; hashes de los originales en `PROVENANCE.json`.
- Cambios de PenaltyStack: `python tools/robot_build.py --check-originals`.
- Compilación: `python tools/robot_build.py`. Despliegue: `DeployCopaNAO.bat`.
- Documentación: `docs/robot/` (auditoría, implementación y operación).

This product includes software developed by B-Human (http://www.b-human.de). Ver
`License.md` y las licencias de cada librería en `Util/`.
