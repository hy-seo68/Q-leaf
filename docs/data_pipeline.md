# Data preparation and construction commands

All paths below are supplied by the caller. Keep data and outputs outside the source tree.

## Vector inputs

Q-LEAF reads float32 fvecs: each row starts with an int32 dimension followed by that many float32 values. Supply base and query files with these dimensions and row counts:

| Dataset | Dimension | Base rows | Query rows | Base / query filenames |
|---|---:|---:|---:|---|
| SIFT10M | 128 | 10,000,000 | 1,000 | `sift10m_base.fvecs` / `sift10m_query.fvecs` |
| DEEP10M | 96 | 9,990,000 | 10,000 | `deep-image-96-angular_base.fvecs` / `deep-image-96-angular_query.fvecs` |
| Text2Image10M | 200 | 10,000,000 | 100,000 | `base.10M.fvecs` / `query.public.100K.fvecs` |
| YFCC10M | 192 | 10,000,000 | 100,000 | `base.10M.fvecs` / `query.public.100K.fvecs` |

DEEP10M is the DEEP1B subset distributed by [ANN-Benchmarks](https://github.com/erikbern/ann-benchmarks) as `deep-image-96-angular.hdf5`. Its 9,990,000 `train` vectors are the base and its 10,000 `test` vectors are the queries.

Text2Image10M is available from the [NeurIPS'21 Billion-Scale ANN dataset page](https://big-ann-benchmarks.com/neurips21.html#datasets); its `.fbin` files store an 8-byte count/dimension header followed by float32 rows.

Supply SIFT10M as preformatted fvecs inputs with the listed row counts.

YFCC10M comes from the [NeurIPS'23 Big-ANN filtered track](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md#datasets). Use `base.10M.u8bin`, `query.public.100K.u8bin`, `base.metadata.10M.spmat`, `query.metadata.public.100K.spmat`, and `GT.public.ibin` from its [dataset loader](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py). Convert uint8 coordinates to float32 without normalization or reordering and write an fvecs dimension prefix for every row. Preserve the official base row IDs so metadata and GT remain aligned. The exporter converts selected public query vectors to float32 and selects single-tag queries for the real-label workload; synthetic YFCC workloads use the full public query set.

Keep vector order consistent with labels, idx mappings, and GT.

## Real-label workload

Run the exporter shown in the README. Then:

```bash
bash scripts/build_index.sh real "$BASE_FVECS" "$EXPORT_DIR/graph_tags.tsv" \
  "$INDEX_DIR" yfcc "$EXPORT_DIR/idx"
GRAPH_DIR="$INDEX_DIR/yfcc/auto_tiers"
bash scripts/compute_eps.sh "$BASE_FVECS" "$GRAPH_DIR" "$GRAPH_DIR/ep_kmeans"
python data_prepare/yfcc_prepare.py config --tags "$EXPORT_DIR/graph_tags.tsv" \
  --graph-dir "$GRAPH_DIR" --ep-dir "$GRAPH_DIR/ep_kmeans" --out "$CONFIG"
```

The real-tag builder copies each label's idx mapping beside its graph. Its medium NN-Descent iteration count defaults to eight; the script explicitly sets `QLEAF_MEDIUM_ITER=8`. Construction uses eight workers, eight large-task threads, at most two concurrent large tasks, and an inclusive 500,000-vector huge-task threshold. Before entry generation, `compute_eps.sh` removes inherited `OMP_`, `OPENBLAS_`, `MKL_`, `NUMEXPR_`, `GOTO_`, `BLIS_`, and `QLEAF_` variables, then restores the explicit construction controls, OMP=8 and OpenBLAS=1; other thread controls such as `OMP_WAIT_POLICY` remain unset.

## Uniform and skewed workloads

Use the generators described in [synthetic preparation](../data_prepare/synthetic/README.md). The construction script partitions an `N 1` base-label file into three temporary label files, preserving row positions and marking other tiers `IGNORE`. Each label's graph uses the size parameters in the README. The driver uses at most eight workers. Synthetic construction sets OMP, OpenBLAS, MKL, and NumExpr thread counts to one with `OMP_DYNAMIC=FALSE`; entry generation uses the separate eight-thread profile above.

```bash
bash scripts/build_index.sh synthetic "$BASE_FVECS" "$BASE_LABELS" "$INDEX_DIR" "$DATASET"
```

Add `--dry-run` as the first argument to write masks plus `commands.json` (commands and environment) without calling either builder. Construction runs large, medium, then small, with one process per nonempty tier and up to eight one-thread graph workers in that process (five workers for five large labels).

For the three tiers, the graph directories under `$INDEX_DIR/$DATASET/` are:

```text
K20_L30_iter5_S10_R50_NSG_L20_R16_C100
K40_L50_iter8_S10_R100_NSG_L30_R32_C100
K150_L150_iter10_S10_R100_NSG_L40_R50_C300
```

For each nonempty tier, run the following with `TIER=small`, `medium`, or `large` and the corresponding `GRAPH_DIR`:

```bash
bash scripts/compute_eps.sh "$BASE_FVECS" "$GRAPH_DIR" "$GRAPH_DIR/ep_kmeans"
python data_prepare/yfcc_prepare.py config --tags "$INDEX_DIR/$TIER.tags.tsv" \
  --graph-dir "$GRAPH_DIR" --ep-dir "$GRAPH_DIR/ep_kmeans" --out "$INDEX_DIR/$TIER.csv"
```

Concatenate the generated tier config files into `$ALL_CONFIG` (each has a comment header). Then prepare the thresholded search config and postings:

```bash
python data_prepare/make_config.py --config "$ALL_CONFIG" --base-labels "$BASE_LABELS" \
  --tau 1000 --out-config "$CONFIG" --postings "$POSTINGS"
```

The config utility handles single nonnegative integer base labels, validates stable postings against the idx mappings, and removes rows with `N_L < tau`. Missing integer tag IDs produce empty CSC columns. Existing outputs are refused. Postings use stable argsort, uint64 indptr, and uint32 IDs; each list is checked against its sorted idx mapping. It does not create graphs or entry points. Config paths are passed unchanged to the C++ reader and interpreted from the process working directory; absolute paths generated from caller-supplied output locations are convenient for moving between working directories.
