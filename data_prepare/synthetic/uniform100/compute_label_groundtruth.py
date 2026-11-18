#!/usr/bin/env python3
"""Prepare compute label groundtruth."""

import numpy as np
import struct
import os
from tqdm import tqdm

def read_fvecs(filename, max_vectors=None):
    """Read fvecs."""
    vectors = []
    with open(filename, 'rb') as f:
        count = 0
        while True:
            if max_vectors is not None and count >= max_vectors:
                break

            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break

            dim = struct.unpack('i', dim_bytes)[0]
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                break

            vec = struct.unpack('f' * dim, vec_bytes)
            vectors.append(vec)
            count += 1

    return np.array(vectors, dtype=np.float32)

def read_label_txt(filename):
    """Read label txt."""
    labels = []
    with open(filename, 'r') as f:

        first_line = f.readline().strip().split()
        num_vectors = int(first_line[0])
        num_labels = int(first_line[1])

        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))

    return np.array(labels, dtype=np.int32)

def write_ivecs(filename, data):
    """Write ivecs."""
    n_vectors, k = data.shape

    with open(filename, 'wb') as f:
        for i in range(n_vectors):

            f.write(struct.pack('i', k))

            for j in range(k):
                f.write(struct.pack('i', int(data[i, j])))

def compute_distance(query, base_vectors):
    """Compute distance."""
    diff = base_vectors - query
    distances_squared = np.sum(diff ** 2, axis=1)
    return distances_squared

def compute_label_groundtruth(base_vectors, query_vectors, base_labels, query_labels, k=10):
    """Compute label groundtruth."""
    n_query = len(query_vectors)
    groundtruth = np.zeros((n_query, k), dtype=np.int32)

    print("\nGrouping base vectors by label...")
    label_to_indices = {}
    for label in np.unique(base_labels):
        label_to_indices[label] = np.where(base_labels == label)[0]

    print("\nBase vectors per label:")
    for label in sorted(label_to_indices.keys())[:10]:
        print(f"  Label {label}: {len(label_to_indices[label]):,}")
    if len(label_to_indices) > 10:
        print(f"  ... (total {len(label_to_indices)} Label)")

    print(f"\nComputing label-constrained KNN for each query...")
    print(f"  - Query count: {n_query:,}")
    print(f"  - K: {k}")

    for i in tqdm(range(n_query), desc="Processing queries"):
        query_label = query_labels[i]
        query_vec = query_vectors[i]

        if query_label not in label_to_indices:
            print(f"\nWarning: Query {i} label {query_label} has no matching base vectors!")

            groundtruth[i] = -1
            continue

        candidate_indices = label_to_indices[query_label]

        if len(candidate_indices) < k:
            print(f"\nWarning: Query {i} label {query_label} has {len(candidate_indices)} vectors available (k={k})")

            actual_k = len(candidate_indices)
        else:
            actual_k = k

        candidate_vectors = base_vectors[candidate_indices]

        distances = compute_distance(query_vec, candidate_vectors)

        nearest_indices = np.argsort(distances)[:actual_k]

        groundtruth[i, :actual_k] = candidate_indices[nearest_indices]

        if actual_k < k:
            groundtruth[i, actual_k:] = -1

    return groundtruth

def main():

    base_file = os.environ["QLEAF_BASE"]
    query_file = os.environ["QLEAF_QUERIES"]
    base_label_file = os.environ["QLEAF_BASE_LABELS"]
    query_label_file = os.environ["QLEAF_QUERY_LABELS"]
    output_file = os.environ["QLEAF_GT"]

    k = 10

    print("=" * 70)
    print("Label-constrained KNN ground truth computation")
    print("=" * 70)

    print("\n[1] Reading label files")
    print("-" * 70)
    print(f"Base labels: reading: {base_label_file}")
    base_labels = read_label_txt(base_label_file)
    print(f"  - Labels read: {len(base_labels):,}")
    print(f"  - Unique labels: {len(np.unique(base_labels))}")

    print(f"\nQuery labels: reading: {query_label_file}")
    query_labels = read_label_txt(query_label_file)
    print(f"  - Labels read: {len(query_labels):,}")
    print(f"  - Unique labels: {len(np.unique(query_labels))}")

    print("\n[2] Reading vector files")
    print("-" * 70)
    print(f"Base vectors: reading: {base_file}")
    print("  (This may take some time...)")
    base_vectors = read_fvecs(base_file)
    print(f"  - Vectors read: {len(base_vectors):,}")
    print(f"  - Vector dimension: {base_vectors.shape[1]}")
    print(f"  - Memory use: {base_vectors.nbytes / (1024**3):.2f} GB")

    print(f"\nQuery vectors: reading: {query_file}")
    query_vectors = read_fvecs(query_file)
    print(f"  - Vectors read: {len(query_vectors):,}")
    print(f"  - Vector dimension: {query_vectors.shape[1]}")

    print("\n[3] Label-constrained KNN ground truth computation")
    print("-" * 70)
    groundtruth = compute_label_groundtruth(
        base_vectors, query_vectors, base_labels, query_labels, k=k
    )

    print("\n[4] Saving results")
    print("-" * 70)
    print(f"Saving: {output_file}")
    write_ivecs(output_file, groundtruth)
    print(f"  - File size: {os.path.getsize(output_file):,} bytes")

    print("\n" + "=" * 70)
    print("Completed!")
    print("=" * 70)
    print(f"\nGenerated files: {output_file}")
    print(f"  - Query count: {len(groundtruth):,}")
    print(f"  - Neighbors per query: {k}")
    print(f"  - Total ground truth pairs: {len(groundtruth) * k:,}")

    print("\nSample results (first 5 queries):")
    for i in range(min(5, len(groundtruth))):
        print(f"  Query {i} (label={query_labels[i]}): {groundtruth[i][:5]}...")
    print()

if __name__ == '__main__':
    main()
