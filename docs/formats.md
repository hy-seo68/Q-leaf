# File formats

Unless stated otherwise, binary integers and vectors use little-endian layout on x86-64. IDs are zero-based. Graph IDs are local to a label; idx maps them to global base-vector IDs. Ground truth and returned search IDs are global.

| File | Layout |
|---|---|
| Base / queries (`.fvecs`) | Each row: int32 dimension, then float32[dimension]. No global header. |
| Ground truth (`.ivecs`) | Each row: int32 neighbor count, then int32 global IDs. The benchmark evaluates the first k entries. |
| Synthetic labels | First line `N 1`, then one label per row. Query `--label-format legacy` skips the header and maps spaces to underscores; `|` separates OR terms. |
| Real labels | One `t<tag>` per row, no header; use `--label-format plain`. |
| Config | Four comma-separated fields: `category,graph_path,idx_path,ep_path`. Blank lines and lines beginning `#` are ignored. Paths are interpreted from the current working directory. An empty or `-` EP field falls back to the NSG navigating node. |
| NSG graph | uint32 width, uint32 navigating local ID; then, for each local node, uint32 degree and uint32[degree] local neighbors. |
| Idx | uint32 count, then uint32[count] global IDs in local-node order. |
| Postings CSC | uint64 ncol, uint64 nnz, uint64[ncol+1] indptr, uint32[nnz] global IDs. |
| Entry points | Comment lines begin `#`; data lines are `local_id \| global_id`. Search reads the local ID. |
| Real graph tags | TSV header `#name`, `tag`, `size`, followed by the label name, numeric tag, and membership count. |
| Query metadata | TSV with `qid`; the benchmark reads available `band`, `path`, and `stratum` fields for grouped summaries. The exporter additionally records `orig_qid`, `tag`, `freq`, and `sigma`. |

The exporter reads official YFCC files as follows:

- u8bin: uint32 number of vectors, uint32 dimension, then uint8[n,dimension]. Exported queries are converted to float32 fvecs.
- spmat: int64 nrow, ncol, nnz; int64[nrow+1] indptr, int32[nnz] indices, and float32[nnz] values. Tag frequency determines graph eligibility.
- GT.public.ibin: uint32 number of queries, uint32 neighbor count, int32[nq,k] global IDs, then float32[nq,k] distances. The exporter copies the matching single-tag ID rows.

The entry generator trains FAISS k-means for 25 iterations with seed 42. FAISS limits training to `min(N_L, 256*m)` samples. Representatives are the full-data vectors nearest to their assigned centroids. Empty final assignments are omitted and ordered duplicates removed, so the saved count can be smaller than m.

Search TSV headers are emitted by `qleaf_bench.cpp`. Entry and search distance counts remain separate. Per-query entry time starts at query setup and ends after entry selection; scan-only queries report zero entry time, and multi-term entry time is NaN. All-group QPS measures the complete parallel query loop; per-query and aggregate output is written afterward. Warmup runs once, before the L/run loops. VmRSS and VmHWM are read from `/proc/self/status` outside the query loop.
