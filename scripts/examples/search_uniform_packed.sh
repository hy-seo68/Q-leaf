#!/usr/bin/env bash
set -euo pipefail
: "${QLEAF_BIN_DIR:?}" "${BASE:?}" "${CONFIG:?}" "${QUERIES:?}" "${LABELS:?}" "${GT:?}" "${OUT:?}"
exec "$QLEAF_BIN_DIR/qleaf_bench" \
  --base "$BASE" \
  --config "$CONFIG" \
  --queries "$QUERIES" \
  --labels "$LABELS" \
  --gt "$GT" \
  --layout packed \
  --visited bitset \
  --packed-record current \
  --threads 1 \
  --k 10 \
  --label-format legacy \
  --init entry \
  --L 10,15,20,25,30,35,40,45,50,55,60,70,90,110,130,160,190,210,240,270,300 \
  --runs 1 \
  --warmup 1000 \
  --out "$OUT"
