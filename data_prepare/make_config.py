#!/usr/bin/env python3
"""Filter numeric-label graphs by size and prepare stable CSC postings."""

import argparse
from pathlib import Path
import numpy as np


def main():
    """Read a graph config and legacy base labels; write a filtered config and CSC."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Input four-column numeric-label config")
    parser.add_argument("--base-labels", required=True, help="Legacy labels with an N 1 header")
    parser.add_argument("--out-config", required=True, help="New filtered config")
    parser.add_argument("--postings", required=True, help="New CSC postings file")
    parser.add_argument("--tau", type=int, default=1000, help="Minimum graph-label size")
    args = parser.parse_args()
    if args.tau < 0:
        parser.error("--tau must be nonnegative")

    cfg = Path(args.config).resolve()
    lines = cfg.read_text().splitlines(keepends=True)
    rows = [line.strip().split(',') for line in lines if line.strip() and not line.startswith('#')]
    tags = [int(row[0]) for row in rows]
    assert len(set(tags)) == len(tags) and tags and min(tags) >= 0
    with Path(args.base_labels).open() as f:
        n, width = map(int, f.readline().split())
        assert width == 1
    labels = np.loadtxt(args.base_labels, dtype=np.int32, skiprows=1)
    labels = np.atleast_1d(labels)
    assert labels.shape == (n,) and labels.min() >= 0
    ncol = max(max(tags), int(labels.max())) + 1
    assert set(tags) == set(np.unique(labels))
    counts = np.bincount(labels, minlength=ncol)
    ptr = np.r_[np.uint64(0), np.cumsum(counts, dtype=np.uint64)].astype('<u8')
    ids = np.argsort(labels, kind='stable').astype('<u4')

    removed = set()
    for name, graph, idx, ep in rows:
        tag = int(name)
        raw = np.fromfile(idx, dtype='<u4')
        nl = int(raw[0])
        assert raw.size == nl + 1 and nl == counts[tag]
        posting = ids[int(ptr[tag]):int(ptr[tag + 1])]
        assert np.array_equal(posting, np.sort(raw[1:])), tag
        if nl < args.tau:
            removed.add(tag)
    kept = [line for line in lines if not line.strip() or line.startswith('#') or int(line.split(',')[0]) not in removed]

    out_cfg = Path(args.out_config).resolve()
    assert not out_cfg.exists() and not Path(args.postings).exists()
    with Path(args.postings).open('xb') as f:
        np.array([ncol, n], dtype='<u8').tofile(f)
        ptr.tofile(f)
        ids.tofile(f)
    with out_cfg.open('x') as f:
        f.writelines(kept)


if __name__ == "__main__":
    main()
