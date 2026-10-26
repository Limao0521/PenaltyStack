#!/bin/sh
set -eu
if command -v pgrep >/dev/null 2>&1 && pgrep -x bhuman >/dev/null 2>&1; then
  echo 'B-Human sigue activo; detenerlo antes de ejecutar PenaltyStack.' >&2
  exit 2
fi
cd /home/nao/PenaltyStack/current
# NAOqi 2.8 trae python2; otras imágenes del robot sólo exponen python3.
# El módulo qi debe existir en el intérprete elegido.
PY=$(command -v python2 || command -v python3 || command -v python)
exec "$PY" -m penalty.nao --profile profile.json "$@"
