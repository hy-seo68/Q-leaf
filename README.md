# Q-LEAF

Q-LEAF performs label-filtered approximate nearest-neighbor search using label-specific NSG graphs and multiple entry points. Disjoint synthetic labels use packed storage; overlapping real labels use a shared vector pool. Labels without a graph are searched exhaustively through postings. This package includes graph construction, entry-point generation, search, and data preparation.

## Requirements and build

Linux on x86-64; GCC 15.2.0 tested, CMake 3.18+, OpenMP, OpenBLAS, Boost headers, and FAISS commit `2135e5a28d0f2a932dd02b64b566ac1a52266540` (CMake project version 1.12.0). NSG and NN-Descent sources are included under `third_party/`; FAISS is external. See [build instructions](docs/build.md) and `environment.yml`.

```bash
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ROOT="$FAISS_ROOT" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build "$BUILD_DIR" --parallel 2
export QLEAF_BIN_DIR="$BUILD_DIR/bin"
```

## Data preparation

YFCC10M uses the official [big-ann-benchmarks filtered track](https://github.com/harsha-simhadri/big-ann-benchmarks/tree/main/benchmark/datasets). The exporter selects single-tag public queries and the corresponding official ground truth. The base used by the C++ tools is the float32 fvecs representation of the official u8bin vectors.

```bash
python data_prepare/yfcc_prepare.py export --base-meta "$BASE_META" \
  --query-meta "$QUERY_META" --query-u8bin "$QUERY_U8BIN" --gt "$OFFICIAL_GT" \
  --out "$EXPORT_DIR" --T 1000
```

Prepare the four datasets as float32 fvecs using the dimensions and vector counts in [vector inputs](docs/data_pipeline.md#vector-inputs). Use [synthetic preparation](data_prepare/synthetic/README.md) for uniform workloads (|L|=12 or |L|=100) and the skewed workload; use the exporter above for the real-label workload.

## Graphs and entry points

| Label size | NN-Descent (K, L, iter, S, R) | NSG (L, R, C) | Entry points |
|---|---|---|---|
| <50,000 | (20, 30, 5, 10, 50) | (20, 16, 100) | 16 |
| 50,000–499,999 | (40, 50, 8, 10, 100) | (30, 32, 100) | 32 |
| ≥500,000 | (150, 150, 10, 10, 100) | (40, 50, 300) | 64 |

Use `scripts/build_index.sh` for synthetic labels or exported real tags, then `scripts/compute_eps.sh` for each graph directory. See [the complete commands](docs/data_pipeline.md). The entry-point script selects FAISS k-means with 25 iterations, seed 42, and automatic 16/32/64 representatives. It sets `OMP_NUM_THREADS=8`, `OPENBLAS_NUM_THREADS=1` and unsets `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `OMP_DYNAMIC`. The underlying executable's defaults remain unchanged, so use the script or explicitly pass `--policy kmeans --num-clusters 0`.

For synthetic workloads, `data_prepare/make_config.py --tau 1000` prepares postings and removes labels smaller than τ from the search config. For real labels, `yfcc_prepare.py export --T 1000` applies the same threshold. These labels use exhaustive scan; τ is not a search executable option.

## Search

Set data paths as environment variables and choose the example for the workload:

- [Uniform workload](scripts/examples/search_uniform_packed.sh): packed storage.
- [Skewed workload](scripts/examples/search_skewed_packed_postings.sh): packed storage with postings and a τ-filtered config.
- [Real-label workload](scripts/examples/search_real_pool_hugepage.sh): shared pool with huge pages.

All three use `--init entry --visited bitset --packed-record current --threads 1 --k 10 --runs 1`.

Inputs are `--base`, `--queries`, `--labels`, `--gt`, and `--config` and/or `--postings`. Select `--label-format legacy` for synthetic labels or `plain` for real labels. Use `--L` for search widths, `--warmup` for the initial query count, `--out` for aggregate TSV, `--meta` for query groups, and `--per-query-out` for per-query TSV. Initialization also supports `--init nsg` and `--init all`. `--runs` repeats within one process; warmup runs once using the first L. All-group QPS uses query-loop wall time; output I/O is outside that interval.

Other options: `--debug-init`, `--result-out`, `--packed-scan-dump`, `--overwrite`, `--visited epoch`, `--packed-record optimized`, `--pool mmap`.

See [file formats](docs/formats.md).

## License and citation

Q-LEAF contributions use the [MIT license](LICENSE). Third-party licenses and notices are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Citation will be added after publication.
