#!/usr/bin/env bash
set -euo pipefail
: "${QLEAF_BIN_DIR:?}" "${BASE:?}" "${CONFIG:?}" "${QUERIES:?}" "${LABELS:?}" "${GT:?}" "${META:?}" "${OUT:?}" "${PER_QUERY_OUT:?}" "${POSTINGS:?}"
exec "$QLEAF_BIN_DIR/qleaf_bench" \
  --base "$BASE" \
  --config "$CONFIG" \
  --queries "$QUERIES" \
  --labels "$LABELS" \
  --gt "$GT" \
  --meta "$META" \
  --layout packed \
  --init entry \
  --visited bitset \
  --packed-record current \
  --k 10 \
  --threads 1 \
  --label-format legacy \
  --prefix '' \
  --L 10,15,20,25,30,35,40,45,50,55,60,70,90,110,130,150,170,190,210 \
  --runs 1 \
  --warmup 10000 \
  --out "$OUT" \
  --per-query-out "$PER_QUERY_OUT" \
  --postings "$POSTINGS"
