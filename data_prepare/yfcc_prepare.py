#!/usr/bin/env python3
"""Prepare yfcc prepare."""
import argparse
import json
import os
import sys
import time

import numpy as np

BAND_NAMES = ["tiny", "sparse", "low", "medium", "large"]
BAND_EDGES = [10 ** -4.5, 10 ** -3.5, 10 ** -2.5, 10 ** -1.5]

def read_spmat(path):
    with open(path, "rb") as f:
        nrow, ncol, nnz = (int(x) for x in np.fromfile(f, dtype=np.int64, count=3))
    off = 24
    indptr = np.memmap(path, dtype=np.int64, mode="r", offset=off, shape=(nrow + 1,))
    off += 8 * (nrow + 1)
    indices = np.memmap(path, dtype=np.int32, mode="r", offset=off, shape=(nnz,))
    if int(indptr[-1]) != nnz:
        raise ValueError(f"{path}: indptr[-1]={int(indptr[-1])} != nnz={nnz}")
    return nrow, ncol, indptr, indices

def read_u8bin(path):
    n, d = (int(x) for x in np.fromfile(path, dtype=np.uint32, count=2))
    expected = 8 + n * d
    if os.path.getsize(path) != expected:
        raise ValueError(f"{path}: size {os.path.getsize(path)} != {expected}")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(n, d))

def read_gt_ibin(path):
    n, k = (int(x) for x in np.fromfile(path, dtype=np.uint32, count=2))
    expected = 8 + n * k * 8
    if os.path.getsize(path) != expected:
        raise ValueError(f"{path}: size {os.path.getsize(path)} != {expected}")
    ids = np.memmap(path, dtype=np.int32, mode="r", offset=8, shape=(n, k))
    dists = np.memmap(path, dtype=np.float32, mode="r", offset=8 + n * k * 4, shape=(n, k))
    return ids, dists

def read_fvecs(path):
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    row_bytes = 4 * (d + 1)
    size = os.path.getsize(path)
    if size % row_bytes:
        raise ValueError(f"{path}: size {size} is not a multiple of row size {row_bytes}")
    m = np.memmap(path, dtype=np.float32, mode="r", shape=(size // row_bytes, d + 1))
    return m[:, 1:]

def read_ivecs(path):
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    a = np.fromfile(path, dtype=np.int32)
    return a.reshape(-1, d + 1)[:, 1:]

def _write_vecs(path, X, dtype):
    X = np.ascontiguousarray(X, dtype=dtype)
    n, d = X.shape
    out = np.empty((n, d + 1), dtype=dtype)
    out[:, 1:] = X
    out.view(np.int32)[:, 0] = d
    out.tofile(path)

def write_fvecs(path, X):
    _write_vecs(path, X, np.float32)

def write_ivecs(path, X):
    _write_vecs(path, X, np.int32)

def write_idx(path, ids):
    ids = np.ascontiguousarray(ids, dtype=np.uint32)
    with open(path, "wb") as f:
        np.array([ids.size], dtype=np.uint32).tofile(f)
        ids.tofile(f)

def write_postings(path, ptr, ids):
    with open(path, "wb") as f:
        np.array([ptr.size - 1, ids.size], dtype=np.uint64).tofile(f)
        np.ascontiguousarray(ptr, dtype=np.uint64).tofile(f)
        np.ascontiguousarray(ids, dtype=np.uint32).tofile(f)

def read_postings(path):
    ncol, nnz = (int(x) for x in np.fromfile(path, dtype=np.uint64, count=2))
    ptr = np.memmap(path, dtype=np.uint64, mode="r", offset=16, shape=(ncol + 1,))
    ids = np.memmap(path, dtype=np.uint32, mode="r", offset=16 + 8 * (ncol + 1), shape=(nnz,))
    return ptr, ids

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def tag_freq(ncol, indices):
    return np.bincount(np.asarray(indices), minlength=ncol).astype(np.int64)

def band_index(sigma):
    return np.digitize(sigma, BAND_EDGES)

def build_csc(nrow, ncol, indptr, indices):
    """Build csc."""
    counts = tag_freq(ncol, indices)
    ptr = np.zeros(ncol + 1, dtype=np.uint64)
    np.cumsum(counts, out=ptr[1:])
    rows = np.repeat(np.arange(nrow, dtype=np.uint32), np.diff(np.asarray(indptr)))
    order = np.argsort(np.asarray(indices), kind="stable")
    return counts, ptr, rows[order]

def topk_bruteforce(base_u8, ids, q, k, chunk=1_000_000):
    """Topk bruteforce."""
    best_d = np.empty(0, dtype=np.float32)
    best_i = np.empty(0, dtype=np.int64)
    q = np.asarray(q, dtype=np.float32)
    for s in range(0, ids.size, chunk):
        sub = ids[s:s + chunk]
        X = np.asarray(base_u8[sub], dtype=np.float32)
        X -= q
        d = np.einsum("ij,ij->i", X, X)
        best_d = np.concatenate([best_d, d])
        best_i = np.concatenate([best_i, sub.astype(np.int64)])
        if best_d.size > k:
            sel = np.argpartition(best_d, k - 1)[:k]
            best_d, best_i = best_d[sel], best_i[sel]
    order = np.lexsort((best_i, best_d))
    return best_i[order][:k], best_d[order][:k]

def cmd_stats(a):
    t0 = time.time()
    nrow, ncol, indptr, indices = read_spmat(a.base_meta)
    nnz = indices.size
    per_vec = np.diff(np.asarray(indptr))
    freq = tag_freq(ncol, indices)
    out = {
        "base": {
            "n": nrow, "vocabulary": ncol, "memberships": int(nnz),
            "tags_per_vector_mean": float(per_vec.mean()),
            "tags_per_vector_max": int(per_vec.max()),
            "vectors_without_tags": int((per_vec == 0).sum()),
            "tags_with_at_least_one_vector": int((freq > 0).sum()),
        }
    }
    dim = a.dim
    rows = []
    for T in [int(x) for x in a.thresholds.split(",")]:
        g = freq >= T
        mem_graph = int(freq[g].sum())
        rows.append({
            "T": T,
            "graph_tags": int(g.sum()),
            "graph_memberships": mem_graph,
            "graph_memberships_over_n": mem_graph / nrow,
            "scan_tags": int(((freq > 0) & ~g).sum()),
            "scan_memberships": int(freq[(freq > 0) & ~g].sum()),
            "replicated_vectors_GB_float32": mem_graph * dim * 4 / 1e9,
            "global_pool_GB_float32": nrow * dim * 4 / 1e9,
        })
    out["thresholds"] = rows
    order = np.argsort(-freq)[: a.top]
    out["largest_tags"] = [{"tag": int(t), "freq": int(freq[t]), "sigma": float(freq[t] / nrow)} for t in order]
    nz = freq > 0
    bidx = band_index(freq[nz] / nrow)
    out["tag_bands"] = {BAND_NAMES[b]: {"tags": int((bidx == b).sum()),
                                        "memberships": int(freq[nz][bidx == b].sum())}
                        for b in range(len(BAND_NAMES))}

    if a.query_meta:
        qn, qncol, qptr, qind = read_spmat(a.query_meta)
        nt = np.diff(np.asarray(qptr))
        comp = {str(int(c)): int((nt == c).sum()) for c in np.unique(nt)}
        single = np.flatnonzero(nt == 1)
        qtag = np.asarray(qind)[np.asarray(qptr)[single]]
        qf = freq[qtag]
        qs = qf / nrow
        qb = band_index(qs)
        qinfo = {
            "queries": qn, "tags_per_query": comp, "single_tag_queries": int(single.size),
            "single_tag_queries_freq_below_k": int((qf < a.k).sum()),
            "single_tag_bands": {BAND_NAMES[b]: int(((qb == b) & (qf >= a.k)).sum()) for b in range(len(BAND_NAMES))},
            "single_tag_path_by_T": {},
        }
        for T in [int(x) for x in a.thresholds.split(",")]:
            ok = qf >= a.k
            qinfo["single_tag_path_by_T"][str(T)] = {"graph": int((ok & (qf >= T)).sum()),
                                                      "scan": int((ok & (qf < T)).sum())}
        out["queries"] = qinfo

    out["elapsed_s"] = time.time() - t0
    text = json.dumps(out, indent=2)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
        log(f"wrote {a.out}")

def cmd_check_base(a):
    U = read_u8bin(a.u8bin)
    F = read_fvecs(a.fvecs)
    if U.shape != F.shape:
        log(f"shape mismatch: u8bin {U.shape} vs fvecs {F.shape}")
        sys.exit(1)
    n = U.shape[0]
    if a.all:
        bad = []
        for s in range(0, n, 1_000_000):
            e = min(n, s + 1_000_000)
            diff = np.any(np.asarray(U[s:e], dtype=np.float32) != np.asarray(F[s:e]), axis=1)
            bad.extend((np.flatnonzero(diff) + s).tolist())
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(a.seed)
        idx = np.unique(np.concatenate([[0, n - 1], rng.integers(0, n, a.samples)]))
        bad = [int(i) for i in idx if not np.array_equal(np.asarray(U[i], dtype=np.float32), np.asarray(F[i]))]
    if bad:
        log(f"MISMATCH in {len(bad)}/{idx.size} sampled rows, e.g. rows {bad[:10]}")
        log("The float base is not row-aligned with base.10M.u8bin: metadata and GT ids would not refer to the same vectors.")
        sys.exit(1)
    log(f"OK: {idx.size} sampled rows identical (shape {U.shape})")

def cmd_export(a):
    t0 = time.time()
    if os.path.exists(a.out):
        if not os.path.isdir(a.out):
            raise ValueError(f"export output exists and is not a directory: {a.out}")
        existing = os.listdir(a.out)
        if existing and not a.overwrite:
            raise ValueError(f"export output directory is not empty: {a.out}; use --overwrite to replace files")
    else:
        os.makedirs(a.out)
    nrow, ncol, indptr, indices = read_spmat(a.base_meta)
    log(f"base metadata: n={nrow}, vocabulary={ncol}, memberships={indices.size}")

    counts, ptr, ids = build_csc(nrow, ncol, indptr, indices)
    write_postings(os.path.join(a.out, "postings.csc"), ptr, ids)
    log("wrote postings.csc")

    gtags = np.flatnonzero(counts >= a.T)
    gtags = gtags[np.argsort(-counts[gtags], kind="stable")]
    idx_dir = os.path.join(a.out, "idx")
    os.makedirs(idx_dir, exist_ok=True)
    with open(os.path.join(a.out, "graph_tags.tsv"), "w") as f:
        f.write("#name\ttag\tsize\n")
        for t in gtags:
            name = f"{a.prefix}{t}"
            write_idx(os.path.join(idx_dir, name + ".idx"), ids[int(ptr[t]):int(ptr[t + 1])])
            f.write(f"{name}\t{t}\t{counts[t]}\n")
    log(f"wrote {gtags.size} graph-tag .idx files (T={a.T}, memberships={int(counts[gtags].sum())})")

    qn, qncol, qptr, qind = read_spmat(a.query_meta)
    if qncol != ncol:
        raise ValueError(f"query vocabulary {qncol} != base vocabulary {ncol}")
    Q = read_u8bin(a.query_u8bin)
    gt_ids, _ = read_gt_ibin(a.gt)
    if Q.shape[0] != qn or gt_ids.shape[0] != qn:
        raise ValueError("query vectors / query metadata / GT row counts differ")
    k = gt_ids.shape[1]
    min_freq = a.min_freq if a.min_freq else k
    nt = np.diff(np.asarray(qptr))
    single = np.flatnonzero(nt == 1)
    qtag = np.asarray(qind)[np.asarray(qptr)[single]].astype(np.int64)
    keep = counts[qtag] >= min_freq
    G = np.asarray(gt_ids[single])
    valid_gt = ((G >= 0) & (G < nrow)).all(axis=1)
    dropped_gt = int((keep & ~valid_gt).sum())
    keep &= valid_gt
    sel, tag = single[keep], qtag[keep]
    f_ = counts[tag]
    sigma = f_ / nrow
    band = band_index(sigma)
    path = np.where(f_ >= a.T, "graph", "scan")

    write_fvecs(os.path.join(a.out, "queries_single.fvecs"), np.asarray(Q[sel], dtype=np.float32))
    write_ivecs(os.path.join(a.out, "gt_single.ivecs"), G[keep])
    with open(os.path.join(a.out, "queries_single.labels.txt"), "w") as f:
        f.writelines(f"{a.prefix}{t}\n" for t in tag)
    with open(os.path.join(a.out, "queries_single.tagids.txt"), "w") as f:
        f.writelines(f"{t + a.label_offset}\n" for t in tag)
    with open(os.path.join(a.out, "queries_single.meta.tsv"), "w") as f:
        f.write("qid\torig_qid\ttag\tfreq\tsigma\tband\tpath\n")
        for i in range(sel.size):
            f.write(f"{i}\t{sel[i]}\t{tag[i]}\t{f_[i]}\t{sigma[i]:.3e}\t{BAND_NAMES[band[i]]}\t{path[i]}\n")
    log(f"single-tag queries: {single.size} in public set, kept {sel.size} "
        f"(dropped {int((~(counts[qtag] >= min_freq)).sum())} with freq<{min_freq}, {dropped_gt} with invalid GT rows)")

    if a.baseline_labels:
        _write_base_labels(os.path.join(a.out, "base_labels.txt"), nrow, indptr, indices, a.label_offset, a.dummy_label)

    summary = {
        "T": a.T, "k": k, "min_freq": min_freq, "graph_tags": int(gtags.size),
        "graph_memberships": int(counts[gtags].sum()), "queries_kept": int(sel.size),
        "queries_by_band": {BAND_NAMES[b]: int((band == b).sum()) for b in range(len(BAND_NAMES))},
        "queries_by_path": {"graph": int((path == "graph").sum()), "scan": int((path == "scan").sum())},
        "elapsed_s": time.time() - t0,
    }
    with open(os.path.join(a.out, "export_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

def _write_base_labels(path, nrow, indptr, indices, offset, dummy, chunk=200_000):
    """ write base labels."""
    ip = np.asarray(indptr)
    empty = 0
    with open(path, "w") as f:
        for r0 in range(0, nrow, chunk):
            r1 = min(nrow, r0 + chunk)
            seg = (np.asarray(indices[ip[r0]:ip[r1]]).astype(np.int64) + offset).astype(str)
            base = ip[r0]
            lines = []
            for r in range(r0, r1):
                a, b = ip[r] - base, ip[r + 1] - base
                if a == b:
                    empty += 1
                    lines.append("" if dummy is None else str(dummy))
                else:
                    lines.append(",".join(seg[a:b]))
            f.write("\n".join(lines) + "\n")
    log(f"wrote {path} ({empty} vectors without tags" + ("" if dummy is None else f", given dummy label {dummy}") + ")")

def cmd_verify_gt(a):
    base = read_u8bin(a.base_u8bin)
    ptr, ids = read_postings(os.path.join(a.export_dir, "postings.csc"))
    Q = read_fvecs(os.path.join(a.export_dir, "queries_single.fvecs"))
    G = read_ivecs(os.path.join(a.export_dir, "gt_single.ivecs"))
    tags = np.loadtxt(os.path.join(a.export_dir, "queries_single.meta.tsv"), dtype=np.int64,
                      skiprows=1, usecols=2, ndmin=1)
    k = G.shape[1]
    rng = np.random.default_rng(a.seed)
    sample = rng.choice(Q.shape[0], size=min(a.samples, Q.shape[0]), replace=False)
    id_match = dist_match = 0
    for i in sample:
        t = int(tags[i])
        plist = np.asarray(ids[int(ptr[t]):int(ptr[t + 1])])
        bi, bd = topk_bruteforce(base, plist, np.asarray(Q[i]), k)
        gd = ((np.asarray(base[G[i]], dtype=np.float64) - np.asarray(Q[i], dtype=np.float64)) ** 2).sum(axis=1)
        if set(bi.tolist()) == set(G[i].tolist()):
            id_match += 1
        if np.allclose(np.sort(gd), bd, rtol=0, atol=1e-6) and np.isin(G[i], plist).all():
            dist_match += 1
    log(f"verify-gt on {sample.size} queries: identical id sets {id_match}, "
        f"distance-consistent (ties allowed) {dist_match}")
    if dist_match != sample.size:
        log("WARNING: GT rows are not reproducible from postings + base; check row alignment and filter semantics.")
        sys.exit(1)

def cmd_or_queries(a):
    base = read_u8bin(a.base_u8bin)
    n = base.shape[0]
    ptr, ids = read_postings(os.path.join(a.export_dir, "postings.csc"))
    counts = np.diff(np.asarray(ptr)).astype(np.int64)
    Q = read_fvecs(os.path.join(a.export_dir, "queries_single.fvecs"))
    rng = np.random.default_rng(a.seed)
    hi_cut = a.hi * n
    hi_tags = np.flatnonzero(counts >= hi_cut)
    lo_tags = np.flatnonzero((counts >= a.min_freq) & (counts < hi_cut))
    pools = {"hh": (hi_tags, hi_tags), "hl": (hi_tags, lo_tags), "ll": (lo_tags, lo_tags)}
    strata = a.strata.split(",")
    per = int(np.ceil(a.n / len(strata)))
    qv, labels, tagids, gts, meta = [], [], [], [], []
    t0 = time.time()
    for s in strata:
        A, B = pools[s]
        if A.size == 0 or B.size == 0 or (s != "hl" and A.size < 2):
            log(f"stratum {s}: not enough tags, skipped")
            continue
        pa = counts[A] / counts[A].sum()
        pb = counts[B] / counts[B].sum()
        made = 0
        while made < per and len(meta) < a.n:
            t1 = int(rng.choice(A, p=pa))
            t2 = int(rng.choice(B, p=pb))
            if t1 == t2:
                continue
            qi = int(rng.integers(Q.shape[0]))
            union = np.union1d(np.asarray(ids[int(ptr[t1]):int(ptr[t1 + 1])]),
                               np.asarray(ids[int(ptr[t2]):int(ptr[t2 + 1])]))
            if union.size < a.k:
                continue
            gi, _ = topk_bruteforce(base, union, np.asarray(Q[qi]), a.k)
            qv.append(np.asarray(Q[qi], dtype=np.float32))
            labels.append(f"{a.prefix}{t1}|{a.prefix}{t2}")
            tagids.append(f"{t1 + a.label_offset},{t2 + a.label_offset}")
            gts.append(gi.astype(np.int32))
            meta.append((len(meta), qi, t1, t2, int(counts[t1]), int(counts[t2]), int(union.size), s))
            made += 1
        log(f"stratum {s}: {made} queries ({time.time() - t0:.0f}s elapsed)")
    d = a.out_dir or a.export_dir
    write_fvecs(os.path.join(d, "or_queries.fvecs"), np.stack(qv))
    write_ivecs(os.path.join(d, "or_gt.ivecs"), np.stack(gts))
    with open(os.path.join(d, "or_queries.labels.txt"), "w") as f:
        f.writelines(x + "\n" for x in labels)
    with open(os.path.join(d, "or_queries.tagids.txt"), "w") as f:
        f.writelines(x + "\n" for x in tagids)
    with open(os.path.join(d, "or_queries.meta.tsv"), "w") as f:
        f.write("qid\tsource_qid\ttag1\ttag2\tfreq1\tfreq2\tunion\tstratum\n")
        for m in meta:
            f.write("\t".join(str(x) for x in m) + "\n")
    log(f"wrote {len(meta)} OR queries to {d}")

def cmd_config(a):
    """Cmd config."""
    idx_dir = a.idx_dir or a.graph_dir
    written, missing = 0, []
    with open(a.out, "w") as f:
        f.write("#category,graph_file,idx_file,entry_points_file\n")
        for line in open(a.tags):
            if line.startswith("#") or not line.strip():
                continue
            name = line.split("\t")[0]
            g = os.path.abspath(os.path.join(a.graph_dir, name + "_nsg.graph"))
            i = os.path.abspath(os.path.join(idx_dir, name + ".idx"))
            e = os.path.abspath(os.path.join(a.ep_dir, name + "_entry_points.txt")) if a.ep_dir else "-"
            if not (os.path.exists(g) and os.path.exists(i) and (e == "-" or os.path.exists(e))):
                missing.append(name)
                continue
            f.write(f"{name},{g},{i},{e}\n")
            written += 1
    log(f"wrote {a.out}: {written} subgraphs" + (f", {len(missing)} missing (e.g. {missing[:5]})" if missing else ""))
    if missing:
        sys.exit(1)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("stats")
    s.add_argument("--base-meta", required=True)
    s.add_argument("--query-meta")
    s.add_argument("--thresholds", default="10,100,500,1000,2000,5000,10000")
    s.add_argument("--dim", type=int, default=192)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--top", type=int, default=20)
    s.add_argument("--out")
    s.set_defaults(func=cmd_stats)

    s = sp.add_parser("check-base")
    s.add_argument("--u8bin", required=True, help="official base.10M.u8bin")
    s.add_argument("--fvecs", required=True, help="float base file used by Q-LEAF")
    s.add_argument("--samples", type=int, default=2000)
    s.add_argument("--all", action="store_true", help="compare every row (reads both files fully, in 1M-row chunks)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_check_base)

    s = sp.add_parser("export")
    s.add_argument("--base-meta", required=True)
    s.add_argument("--query-meta", required=True)
    s.add_argument("--query-u8bin", required=True)
    s.add_argument("--gt", required=True, help="official GT.public.ibin")
    s.add_argument("--out", required=True)
    s.add_argument("--T", type=int, default=1000, help="tags with freq >= T get a graph; smaller tags are scanned")
    s.add_argument("--min-freq", type=int, default=0, help="drop queries whose tag has fewer vectors (default: k)")
    s.add_argument("--prefix", default="t", help="label name prefix used in .idx names and query label files")
    s.add_argument("--baseline-labels", action="store_true", help="also write base_labels.txt for baselines")
    s.add_argument("--label-offset", type=int, default=0, help="added to tag ids in baseline label files")
    s.add_argument("--dummy-label", type=int, default=None, help="label written for vectors without tags")
    s.add_argument("--overwrite", action="store_true", help="allow replacing files in a non-empty output directory")
    s.set_defaults(func=cmd_export)

    s = sp.add_parser("verify-gt")
    s.add_argument("--base-u8bin", required=True)
    s.add_argument("--export-dir", required=True)
    s.add_argument("--samples", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_verify_gt)

    s = sp.add_parser("or-queries")
    s.add_argument("--base-u8bin", required=True)
    s.add_argument("--export-dir", required=True)
    s.add_argument("--out-dir")
    s.add_argument("--n", type=int, default=1000)
    s.add_argument("--strata", default="hh,hl,ll", help="hh: both frequent, hl: mixed, ll: both infrequent")
    s.add_argument("--hi", type=float, default=1e-3, help="selectivity separating frequent from infrequent tags")
    s.add_argument("--min-freq", type=int, default=10)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--prefix", default="t")
    s.add_argument("--label-offset", type=int, default=0)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_or_queries)

    s = sp.add_parser("config")
    s.add_argument("--tags", required=True, help="graph_tags.tsv from export")
    s.add_argument("--graph-dir", required=True, help="directory with <name>_nsg.graph")
    s.add_argument("--idx-dir", help="directory with <name>.idx (default: graph-dir)")
    s.add_argument("--ep-dir", help="directory with <name>_entry_points.txt (omit to use the NSG navigating node)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_config)

    a = p.parse_args()
    a.func(a)

if __name__ == "__main__":
    main()
