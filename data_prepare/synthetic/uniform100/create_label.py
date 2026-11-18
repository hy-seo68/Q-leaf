#!/usr/bin/env python3
"""Prepare create label."""

import numpy as np
import struct
import os

def read_fvecs(filename):
    """Read fvecs."""
    with open(filename, 'rb') as f:

        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            raise ValueError(f"Empty or corrupted file: {filename}")

        dim = struct.unpack('i', dim_bytes)[0]

        f.seek(0, 2)
        file_size = f.tell()

        bytes_per_vector = 4 + dim * 4
        num_vectors = file_size // bytes_per_vector

        print(f"File: {filename}")
        print(f"  - Vectors: {num_vectors:,}")
        print(f"  - Dimension: {dim}")

        return num_vectors, dim

def write_label_txt(filename, data):
    """Write label txt."""
    n_vectors = len(data)
    num_labels = 1

    with open(filename, 'w') as f:

        f.write(f"{n_vectors} {num_labels}\n")

        for i in range(n_vectors):
            f.write(f"{int(data[i])}\n")

    print(f"\nSaved: {filename}")
    print(f"  - Vectors: {n_vectors:,}")
    print(f"  - Labels per vector: {num_labels}")

def generate_uniform_labels(num_vectors, num_classes=100):
    """Generate uniform labels."""
    print(f"\nGenerating labels...")
    print(f"  - Total classes: {num_classes} (0~{num_classes-1})")
    print(f"  - Labels per vector: 1")
    print(f"  - Distribution: uniform")

    labels = np.array([i % num_classes for i in range(num_vectors)], dtype=np.int32)

    np.random.shuffle(labels)

    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel distribution statistics:")
    print(f"  - Minimum count: {counts.min():,}")
    print(f"  - Maximum count: {counts.max():,}")
    print(f"  - Mean count: {counts.mean():.2f}")
    print(f"  - Standard deviation: {counts.std():.2f}")

    return labels

def main():

    base_file = os.environ["QLEAF_BASE"]
    query_file = os.environ["QLEAF_QUERIES"]

    output_dir = os.environ["QLEAF_OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    base_label_file = os.path.join(output_dir, 'label_sift10m_base.txt')
    query_label_file = os.path.join(output_dir, 'label_sift10m_query.txt')

    print("=" * 60)
    print("SIFT10M label ground truth generation")
    print("Assign labels uniformly in the range 0-99")
    print("=" * 60)

    print("\n[1] Base Dataset processing")
    print("-" * 60)
    num_base_vectors, base_dim = read_fvecs(base_file)

    np.random.seed(42)
    base_labels = generate_uniform_labels(num_base_vectors, num_classes=100)
    write_label_txt(base_label_file, base_labels)

    print("\n[2] Query Dataset processing")
    print("-" * 60)
    num_query_vectors, query_dim = read_fvecs(query_file)

    np.random.seed(123)
    query_labels = generate_uniform_labels(num_query_vectors, num_classes=100)
    write_label_txt(query_label_file, query_labels)

    print("\n" + "=" * 60)
    print("Completed!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  1. {base_label_file}")
    print(f"     - File size: {os.path.getsize(base_label_file):,} bytes")
    print(f"  2. {query_label_file}")
    print(f"     - File size: {os.path.getsize(query_label_file):,} bytes")
    print()

if __name__ == '__main__':
    main()
