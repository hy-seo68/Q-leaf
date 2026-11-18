#!/usr/bin/env python3
"""Prepare create label12 dataset."""

import numpy as np
import struct
import os
from collections import defaultdict
from tqdm import tqdm

NUM_LABELS = 100
BASE_SIZE = 9990000
QUERY_SIZE = 10000
K = 10
BATCH_SIZE = 1000

SOURCE_DIR = os.environ["QLEAF_SOURCE_DIR"]
TARGET_DIR = os.environ["QLEAF_OUTPUT_DIR"]
LABEL_DIR = os.path.join(TARGET_DIR, "deep10m_label")

BASE_FVECS = os.path.join(SOURCE_DIR, "deep-image-96-angular_base.fvecs")
QUERY_FVECS = os.path.join(SOURCE_DIR, "deep-image-96-angular_query.fvecs")

OUTPUT_BASE_LABEL = os.path.join(LABEL_DIR, "label_deep10m_base.txt")
OUTPUT_QUERY_LABEL = os.path.join(LABEL_DIR, "label_deep10m_query.txt")
OUTPUT_GROUNDTRUTH = os.path.join(TARGET_DIR, "deep10m_label_groundtruth.ivecs")

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

def compute_label_groundtruth_batch(base_vectors, query_vectors, base_labels, query_labels, k=100, batch_size=BATCH_SIZE):
    """Compute label groundtruth batch."""
    print(f"\nComputing label-filtered groundtruth (k={k})...")
    print(f"Using vectorized batch processing (batch_size={batch_size})")

    num_queries = len(query_vectors)
    groundtruth = np.full((num_queries, k), -1, dtype=np.int32)

    print("Building label index...")
    label_to_indices = {}

    for label in np.unique(base_labels):
        indices = np.where(base_labels == label)[0]
        label_to_indices[label] = indices

    print(f"\nLabel distribution in base:")
    for label in sorted(label_to_indices.keys()):
        print(f"  Label {label}: {len(label_to_indices[label]):,} vectors")

    print("\nGrouping queries by label...")
    query_groups = defaultdict(list)
    for q_idx, label in enumerate(query_labels):
        query_groups[label].append(q_idx)

    print(f"\nLabel distribution in queries:")
    for label in sorted(query_groups.keys()):
        print(f"  Label {label}: {len(query_groups[label]):,} queries")

    print(f"\nProcessing queries by label groups...")
    total_processed = 0

    with tqdm(total=num_queries, desc="Processing queries") as pbar:
        for label in sorted(query_groups.keys()):
            query_indices = query_groups[label]

            if label not in label_to_indices:
                pbar.update(len(query_indices))
                continue

            candidate_indices = label_to_indices[label]
            candidate_vectors = base_vectors[candidate_indices]
            actual_k = min(k, len(candidate_indices))

            num_queries_in_label = len(query_indices)

            for batch_start in range(0, num_queries_in_label, batch_size):
                batch_end = min(batch_start + batch_size, num_queries_in_label)
                batch_query_indices = query_indices[batch_start:batch_end]
                batch_queries = query_vectors[batch_query_indices]

                query_norms = np.sum(batch_queries ** 2, axis=1, keepdims=True)
                candidate_norms = np.sum(candidate_vectors ** 2, axis=1, keepdims=True).T
                dot_products = batch_queries @ candidate_vectors.T

                distances = query_norms + candidate_norms - 2 * dot_products

                if actual_k < len(candidate_indices):

                    partitioned_indices = np.argpartition(distances, actual_k - 1, axis=1)[:, :actual_k]

                    batch_indices = np.arange(len(batch_query_indices))[:, np.newaxis]
                    k_distances = distances[batch_indices, partitioned_indices]

                    sorted_positions = np.argsort(k_distances, axis=1)
                    nearest_indices = partitioned_indices[batch_indices, sorted_positions]
                else:

                    nearest_indices = np.argsort(distances, axis=1)[:, :actual_k]

                for i, q_idx in enumerate(batch_query_indices):
                    groundtruth[q_idx, :actual_k] = candidate_indices[nearest_indices[i]]

                pbar.update(len(batch_query_indices))

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
    print("Deep10M Label12 Dataset Generator")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Number of labels: {NUM_LABELS} (0 to {NUM_LABELS-1})")
    print(f"  Base size: {BASE_SIZE:,}")
    print(f"  Query size: {QUERY_SIZE:,}")
    print(f"  Groundtruth k: {K}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Source directory: {SOURCE_DIR}")
    print(f"  Target directory: {TARGET_DIR}")
    print()

    print("Checking input files...")
    if not os.path.exists(BASE_FVECS):
        raise FileNotFoundError(f"Base file not found: {BASE_FVECS}")
    if not os.path.exists(QUERY_FVECS):
        raise FileNotFoundError(f"Query file not found: {QUERY_FVECS}")
    print(f"  Base file: {BASE_FVECS} ({os.path.getsize(BASE_FVECS):,} bytes)")
    print(f"  Query file: {QUERY_FVECS} ({os.path.getsize(QUERY_FVECS):,} bytes)")

    os.makedirs(TARGET_DIR, exist_ok=True)
    os.makedirs(LABEL_DIR, exist_ok=True)

    print("\nStep 1: Generating labels...")
    print(f"  Generating {BASE_SIZE:,} base labels...")
    base_labels = generate_labels(BASE_SIZE, NUM_LABELS, seed=42)
    print(f"  Generating {QUERY_SIZE:,} query labels...")
    query_labels = generate_labels(QUERY_SIZE, NUM_LABELS, seed=123)

    print("\nLabel distribution (base):")
    unique, counts = np.unique(base_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count:,} ({count/BASE_SIZE*100:.2f}%)")

    print("\nLabel distribution (query):")
    unique, counts = np.unique(query_labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count:,} ({count/QUERY_SIZE*100:.2f}%)")

    print("\nStep 2: Saving labels...")
    save_labels(base_labels, OUTPUT_BASE_LABEL)
    save_labels(query_labels, OUTPUT_QUERY_LABEL)

    print("\nStep 3: Reading vectors...")
    base_vectors = read_fvecs(BASE_FVECS, max_count=BASE_SIZE)
    query_vectors = read_fvecs(QUERY_FVECS, max_count=QUERY_SIZE)

    print(f"Base vectors shape: {base_vectors.shape}")
    print(f"Query vectors shape: {query_vectors.shape}")

    base_memory_gb = base_vectors.nbytes / (1024**3)
    query_memory_gb = query_vectors.nbytes / (1024**3)
    print(f"\nMemory usage:")
    print(f"  Base vectors: {base_memory_gb:.2f} GB")
    print(f"  Query vectors: {query_memory_gb:.2f} GB")
    print(f"  Total: {base_memory_gb + query_memory_gb:.2f} GB")

    print("\nStep 4: Computing label-filtered groundtruth...")
    groundtruth = compute_label_groundtruth_batch(
        base_vectors, query_vectors,
        base_labels, query_labels,
        k=K
    )

    invalid_count = np.sum(groundtruth == -1)
    if invalid_count > 0:
        print(f"\nWarning: {invalid_count} groundtruth entries are -1 (insufficient candidates)")
        print(f"  This means some queries had fewer than {K} matching base vectors")

    print("\nStep 5: Saving groundtruth...")
    save_ivecs(groundtruth, OUTPUT_GROUNDTRUTH)

    print("\n" + "=" * 80)
    print("Dataset generation completed successfully!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  Base labels: {OUTPUT_BASE_LABEL}")
    print(f"  Query labels: {OUTPUT_QUERY_LABEL}")
    print(f"  Groundtruth: {OUTPUT_GROUNDTRUTH}")
    print()

    print("Verification:")
    print(f"  Base labels file size: {os.path.getsize(OUTPUT_BASE_LABEL):,} bytes")
    print(f"  Query labels file size: {os.path.getsize(OUTPUT_QUERY_LABEL):,} bytes")
    print(f"  Groundtruth file size: {os.path.getsize(OUTPUT_GROUNDTRUTH):,} bytes")
    print()

if __name__ == "__main__":
    main()
