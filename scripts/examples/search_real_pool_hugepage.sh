#!/usr/bin/env bash
set -euo pipefail
: "${QLEAF_BIN_DIR:?}" "${BASE:?}" "${CONFIG:?}" "${QUERIES:?}" "${LABELS:?}" "${GT:?}" "${OUT:?}" "${META:?}" "${POSTINGS:?}" "${PER_QUERY_OUT:?}"
exec "$QLEAF_BIN_DIR/qleaf_bench" \
  --base "$BASE" \
  --config "$CONFIG" \
  --queries "$QUERIES" \
  --labels "$LABELS" \
  --gt "$GT" \
  --layout pool \
  --visited bitset \
  --packed-record current \
  --threads 1 \
  --k 10 \
  --label-format plain \
  --init entry \
  --L 15,20,25,30 \
  --runs 1 \
  --warmup 61626 \
  --out "$OUT" \
  --meta "$META" \
  --postings "$POSTINGS" \
  --pool hugepage \
  --per-query-out "$PER_QUERY_OUT"
