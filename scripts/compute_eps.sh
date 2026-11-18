#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
  printf 'Usage: %s <base.fvecs> <graph-dir> <new-ep-dir>\n' "$0" >&2
  exit 64
fi
: "${QLEAF_BIN_DIR:?Set QLEAF_BIN_DIR to the CMake build bin directory}"
ep_binary="$QLEAF_BIN_DIR/compute_ep_nsg"
while IFS= read -r name; do
  case $name in
    OMP_*|OPENBLAS_*|MKL_*|NUMEXPR_*|GOTO_*|BLIS_*|QLEAF_*) unset "$name" ;;
  esac
done < <(compgen -e)
export TZ=Asia/Seoul LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
export QLEAF_BUILD_THREADS=8 QLEAF_LARGE_THREADS=8 QLEAF_MAX_LARGE=2
export QLEAF_HUGE_MIN_SIZE=500000 QLEAF_MEDIUM_ITER=8 QLEAF_KEEP_KNNG=0 QLEAF_OVERWRITE=0
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
exec "$ep_binary" --base-fvecs "$1" --graph-dir "$2" \
  --output-dir "$3" --num-clusters 0 --policy kmeans
