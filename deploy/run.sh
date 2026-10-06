#!/bin/sh
set -eu
if command -v pgrep >/dev/null 2>&1 && pgrep -x bhuman >/dev/null 2>&1; then
  echo 'B-Human sigue activo; detenerlo antes de ejecutar PenaltyStack.' >&2
  exit 2
fi
cd /home/nao/PenaltyStack/current
exec python2 -m penalty.nao --profile profile.json
