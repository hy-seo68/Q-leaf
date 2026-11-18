#!/usr/bin/env python3
import numpy as np
import struct
import os
from collections import defaultdict
from tqdm import tqdm

NUM_LABELS = 12
BASE_SIZE = 10000000
QUERY_SIZE = 1000
K = 10

SOURCE_DIR = os.environ["QLEAF_SOURCE_DIR"]
TARGET_DIR = os.environ["QLEAF_OUTPUT_DIR"]
LABEL_DIR = os.path.join(TARGET_DIR, "sift_label")

BASE_FVECS = os.path.join(SOURCE_DIR, "sift10m_base.fvecs")
QUERY_FVECS = os.path.join(SOURCE_DIR, "sift10m_query.fvecs")

OUTPUT_BASE_LABEL = os.path.join(LABEL_DIR, "label_sift10m_base.txt")
OUTPUT_QUERY_LABEL = os.path.join(LABEL_DIR, "label_sift10m_query.txt")
OUTPUT_GROUNDTRUTH = os.path.join(TARGET_DIR, "sift10m_label_groundtruth.ivecs")

def read_fvecs(filename, max_count=None):
    """Read fvecs."""
    print(f"Reading {filename}...")
    vectors = []
    with open(filename, 'rb') as f:
        count = 0
        while True:
            if max_count and count >= max_count:
                break

            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            dim = struct.unpack('i', dim_bytes)[0]

            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(np.array(vec, dtype=np.float32))
            count += 1

            if count % 100000 == 0:
                print(f"  Read {count} vectors...")

    print(f"Finished reading {len(vectors)} vectors")
    return np.array(vectors, dtype=np.float32)

def generate_labels(size, num_labels, seed=42):
    """Generate labels."""
    print(f"  Generating labels with uniform distribution...")
    print(f"    - Total classes: {num_labels} (0~{num_labels-1})")
    print(f"    - Distribution method: Uniform distribution")

    np.random.seed(seed)

    labels = np.array([i % num_labels for i in range(size)], dtype=np.int32)
    np.random.shuffle(labels)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"    - Min count: {counts.min():,}")
    print(f"    - Max count: {counts.max():,}")
    print(f"    - Mean count: {counts.mean():.2f}")
    print(f"    - Std dev: {counts.std():.2f}")

    return labels

def save_labels(labels, filename):
    """Save labels."""
    print(f"Saving labels to {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    n_vectors = len(labels)
    num_labels = 1

    with open(filename, 'w') as f:

        f.write(f"{n_vectors} {num_labels}\n")

        for label in labels:
            f.write(f"{label}\n")

    print(f"Saved {len(labels)} labels")

def compute_distance(query, base_vectors):
    diff = base_vectors - query
    distances_squared = np.sum(diff ** 2, axis=1)
    return distances_squared

def compute_label_groundtruth(base_vectors, query_vectors, base_labels, query_labels, k=100):
    """Compute label groundtruth."""
    print(f"\nComputing label-filtered groundtruth (k={k})...")
    num_queries = len(query_vectors)
    groundtruth = np.zeros((num_queries, k), dtype=np.int32)

    print("Building label index...")
    label_to_indices = {}
    for label in np.unique(base_labels):
        label_to_indices[label] = np.where(base_labels == label)[0]

    print(f"\nLabel distribution in base:")
    for label in sorted(label_to_indices.keys()):
        print(f"  Label {label}: {len(label_to_indices[label]):,} vectors")

    print(f"\nProcessing each query for label-constrained KNN...")
    print(f"  - Query count: {num_queries:,}")
    print(f"  - K: {k}")

    for q_idx in tqdm(range(num_queries), desc="Processing queries"):
        query_vec = query_vectors[q_idx]
        query_label = query_labels[q_idx]

        if query_label not in label_to_indices:
            print(f"\nWarning: Query {q_idx} with label {query_label} has no matching base vectors!")
            groundtruth[q_idx] = -1
            continue

        candidate_indices = label_to_indices[query_label]

        if len(candidate_indices) < k:
            print(f"\nWarning: Query {q_idx} label {query_label} has only {len(candidate_indices)} vectors (k={k})")
            actual_k = len(candidate_indices)
        else:
            actual_k = k

        candidate_vectors = base_vectors[candidate_indices]

        distances = compute_distance(query_vec, candidate_vectors)

        nearest_indices = np.argsort(distances)[:actual_k]

        groundtruth[q_idx, :actual_k] = candidate_indices[nearest_indices]

        if actual_k < k:
            groundtruth[q_idx, actual_k:] = -1

    print("Groundtruth computation completed")
    return groundtruth

def save_ivecs(data, filename):
    """Save ivecs."""
    print(f"Saving groundtruth to {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        for row in data:
            dim = len(row)
            f.write(struct.pack('i', dim))
            f.write(struct.pack('i' * dim, *row))

    print(f"Saved groundtruth for {len(data)} queries")

def main():
    print("=" * 80)
    print("SIFT10M Label10 Dataset Generator")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Number of labels: {NUM_LABELS} (0 to {NUM_LABELS-1})")
    print(f"  Base size: {BASE_SIZE}")
    print(f"  Query size: {QUERY_SIZE}")
    print(f"  Groundtruth k: {K}")
    print(f"  Source directory: {SOURCE_DIR}")
    print(f"  Target directory: {TARGET_DIR}")
    print()

    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    print("Step 1: Generating labels...")
    print(f"  Generating {BASE_SIZE} base labels...")
    base_labels = generate_labels(BASE_SIZE, NUM_LABELS, seed=42)
    print(f"  Generating {QUERY_SIZE} query labels...")
    query_labels = generate_labels(QUERY_SIZE, NUM_LABELS, seed=123)

    print("\nLabel distribution (base):")
    unique, counts = np.unique(base_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} ({count/BASE_SIZE*100:.2f}%)")

    print("\nLabel distribution (query):")
    unique, counts = np.unique(query_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count} ({count/QUERY_SIZE*100:.2f}%)")

    print("\nStep 2: Saving labels...")
    save_labels(base_labels, OUTPUT_BASE_LABEL)
    save_labels(query_labels, OUTPUT_QUERY_LABEL)

    print("\nStep 3: Reading vectors...")
    base_vectors = read_fvecs(BASE_FVECS, max_count=BASE_SIZE)
    query_vectors = read_fvecs(QUERY_FVECS, max_count=QUERY_SIZE)

    print(f"Base vectors shape: {base_vectors.shape}")
    print(f"Query vectors shape: {query_vectors.shape}")

    print("\nStep 4: Computing label-filtered groundtruth...")
    groundtruth = compute_label_groundtruth(
        base_vectors, query_vectors,
        base_labels, query_labels,
        k=K
    )

    print("\nStep 5: Saving groundtruth...")
    save_ivecs(groundtruth, OUTPUT_GROUNDTRUTH)

    print("\nStep 6: Creating symbolic links to original fvecs files...")
    import subprocess

    target_base_fvecs = os.path.join(TARGET_DIR, "sift10m_base.fvecs")
    target_query_fvecs = os.path.join(TARGET_DIR, "sift10m_query.fvecs")

    subprocess.run(f"rm -f {target_base_fvecs}", shell=True)
    subprocess.run(f"rm -f {target_query_fvecs}", shell=True)

    subprocess.run(f"ln -s {BASE_FVECS} {target_base_fvecs}", shell=True)
    subprocess.run(f"ln -s {QUERY_FVECS} {target_query_fvecs}", shell=True)

    print(f"Created symbolic link: {target_base_fvecs} -> {BASE_FVECS}")
    print(f"Created symbolic link: {target_query_fvecs} -> {QUERY_FVECS}")

    print("\n" + "=" * 80)
    print("Dataset generation completed successfully!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  Base labels: {OUTPUT_BASE_LABEL}")
    print(f"  Query labels: {OUTPUT_QUERY_LABEL}")
    print(f"  Groundtruth: {OUTPUT_GROUNDTRUTH}")
    print(f"  Base vectors: {target_base_fvecs} (symlink)")
    print(f"  Query vectors: {target_query_fvecs} (symlink)")
    print()

    print("Verification:")
    print(f"  Base labels file size: {os.path.getsize(OUTPUT_BASE_LABEL)} bytes")
    print(f"  Query labels file size: {os.path.getsize(OUTPUT_QUERY_LABEL)} bytes")
    print(f"  Groundtruth file size: {os.path.getsize(OUTPUT_GROUNDTRUTH)} bytes")
    print()

if __name__ == "__main__":
    main()
