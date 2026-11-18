#!/usr/bin/env bash
set -euo pipefail
dry_run=0
if [[ ${1:-} == --dry-run ]]; then
  dry_run=1
  shift
fi
if [[ $# -lt 5 || $# -gt 6 || ( $1 != synthetic && $1 != real ) || ( $1 == real && $# -ne 6 ) ]]; then
  printf 'Usage: %s synthetic <base.fvecs> <base-labels.txt> <new-output-dir> <dataset>\n' "$0" >&2
  printf '       %s real <base.fvecs> <graph_tags.tsv> <new-output-dir> <dataset> <idx-dir>\n' "$0" >&2
  printf 'Add --dry-run first to write masks and command/environment records only.\n' >&2
  exit 64
fi
: "${QLEAF_BIN_DIR:?Set QLEAF_BIN_DIR to the CMake build bin directory}"
PYTHON=${PYTHON:-python3}
mode=$1 base=$2 input=$3 output=$4 dataset=$5 idx=${6:-}
if [[ -e $output || -L $output ]]; then
  printf 'Refusing existing output: %s\n' "$output" >&2
  exit 73
fi
mkdir -p -- "$output"
export TZ=Asia/Seoul LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
export QLEAF_BUILD_THREADS=8 QLEAF_LARGE_THREADS=8 QLEAF_MAX_LARGE=2
export QLEAF_HUGE_MIN_SIZE=500000 QLEAF_MEDIUM_ITER=8 QLEAF_KEEP_KNNG=0 QLEAF_OVERWRITE=0
record_command() {
  "$PYTHON" - "$output" "$@" <<'PY'
import json
import os
from pathlib import Path
import sys
out = Path(sys.argv[1]) / 'commands.json'
records = json.loads(out.read_text()) if out.exists() else []
prefixes = ('OMP_', 'OPENBLAS_', 'MKL_', 'NUMEXPR_', 'QLEAF_', 'GOTO_', 'BLIS_')
records.append({'argv': sys.argv[2:],
                'environment': {k: v for k, v in os.environ.items()
                                if k.startswith(prefixes) or k in ('TZ', 'LC_ALL', 'PYTHONDONTWRITEBYTECODE')}})
out.write_text(json.dumps(records, indent=2) + '\n')
PY
}
if [[ $mode == real ]]; then
  export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
  record_command "$QLEAF_BIN_DIR/build_nsg_subgraphs_ml" "$base" "$input" "$idx" "$output" "$dataset"
  [[ $dry_run == 0 ]] || exit 0
  exec "$QLEAF_BIN_DIR/build_nsg_subgraphs_ml" "$base" "$input" "$idx" "$output" "$dataset"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE
unset OMP_WAIT_POLICY

# Preserve row positions while selecting labels for each construction tier.
"$PYTHON" - "$input" "$output" <<'PY'
from collections import Counter
from pathlib import Path
import sys
source, out = map(Path, sys.argv[1:])
with source.open() as f:
    header = f.readline()
    n, width = map(int, header.split())
    assert width == 1
    labels = [line.rstrip('\n') for line in f]
assert len(labels) == n
counts = Counter(labels)
groups = {label: ('small' if count < 50000 else 'medium' if count < 500000 else 'large')
          for label, count in counts.items()}
for tier in ('small', 'medium', 'large'):
    if tier not in groups.values():
        continue
    with (out / (tier + '.labels')).open('x') as f:
        f.write(header)
        f.writelines((label if groups[label] == tier else 'IGNORE') + '\n' for label in labels)
    with (out / (tier + '.tags.tsv')).open('x') as f:
        f.write('#name\ttag\tsize\n')
        for label in sorted(counts):
            if groups[label] == tier:
                f.write(f'{label}\t{label}\t{counts[label]}\n')
PY
for tier in large medium small; do
  [[ -f $output/$tier.labels ]] || continue
  case $tier in
    small) params=(20 30 5 10 50 20 16 100) ;;
    medium) params=(40 50 8 10 100 30 32 100) ;;
    large) params=(150 150 10 10 100 40 50 300) ;;
  esac
  record_command "$QLEAF_BIN_DIR/build_nsg_subgraphs" "$base" "$output/$tier.labels" "$output" "$dataset" "${params[@]}"
  if [[ $dry_run == 0 ]]; then
    "$QLEAF_BIN_DIR/build_nsg_subgraphs" "$base" "$output/$tier.labels" "$output" "$dataset" "${params[@]}"
  fi
done
