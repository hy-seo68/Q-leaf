# Synthetic workloads

Use uniform workloads with \|L\|=12 or \|L\|=100, or the skewed workload with 235 labels. Labels use the legacy `N 1` header followed by one integer label per vector. Base/query vectors and labels must have matching row counts.

Paths below are relative to this directory. Set the input/output variables before running each script.

| Workload | Dataset | Generator | Arguments / path variables |
|---|---|---|---|
| Uniform, \|L\|=12 | SIFT10M | `uniform12/sift/create_label_dataset.py` | No CLI arguments; `QLEAF_SOURCE_DIR`, `QLEAF_OUTPUT_DIR` |
| Uniform, \|L\|=12 | DEEP10M | `generate_uniform_labels.py` | `--dataset deep10m --num-labels 12 --seed 42 --skip-gt`; `QLEAF_DATASET_DIR` |
| Uniform, \|L\|=12 | Text2Image10M | `generate_uniform_labels.py` | `--dataset text2image10m --num-labels 12 --seed 42 --skip-gt`; `QLEAF_DATASET_DIR` |
| Uniform, \|L\|=12 | YFCC10M | `generate_uniform_labels.py` | `--dataset yfcc10m --num-labels 12 --seed 42 --skip-gt`; `QLEAF_DATASET_DIR` |
| Uniform, \|L\|=100 | SIFT10M | `uniform100/create_label.py` | No CLI arguments; `QLEAF_BASE`, `QLEAF_QUERIES`, `QLEAF_OUTPUT_DIR` |
| Uniform, \|L\|=100 | DEEP10M | `uniform100/deep/create_label12_dataset.py` | No CLI arguments; `QLEAF_SOURCE_DIR`, `QLEAF_OUTPUT_DIR` |
| Uniform, \|L\|=100 | Text2Image10M | `uniform100/text2image/create_label12_dataset.py` | No CLI arguments; `QLEAF_SOURCE_DIR`, `QLEAF_OUTPUT_DIR` |
| Uniform, \|L\|=100 | YFCC10M | `generate_uniform_labels.py` | `--dataset yfcc10m --num-labels 100 --seed 42 --skip-gt`; `QLEAF_DATASET_DIR` |
| Skewed | SIFT10M | `generate_segmented_labels_all.py` | `--dataset sift10m --seed 42 --skip-gt`; `QLEAF_DATASET_ROOT`, `QLEAF_OUTPUT_ROOT` |
| Skewed | DEEP10M | `generate_segmented_labels_all.py` | `--dataset deep10m --seed 42 --skip-gt`; `QLEAF_DATASET_ROOT`, `QLEAF_OUTPUT_ROOT` |
| Skewed | Text2Image10M | `generate_segmented_labels_all.py` | `--dataset text2image10m --seed 42 --skip-gt`; `QLEAF_DATASET_ROOT`, `QLEAF_OUTPUT_ROOT` |
| Skewed | YFCC10M | `generate_segmented_labels_all.py` | `--dataset yfcc10m --seed 42 --skip-gt`; `QLEAF_DATASET_ROOT`, `QLEAF_OUTPUT_ROOT` |

Uniform labels use balanced modulo assignment and shuffling, with base seed 42 and query seed 123. `generate_uniform_labels.py` defaults to 12 labels, seed 42, and k=10; its query seed is `--seed + 81`. Under `QLEAF_DATASET_DIR`, use the subdirectories `sift10m`, `deep10m`, `text2image10m`, and `yfcc10m` with the filenames in [vector inputs](../../docs/data_pipeline.md#vector-inputs). It writes `<dataset>_label<num-labels>/base_labels.txt` and `query_labels.txt` and links the input vectors there. `--output-suffix` changes the output directory suffix.

The DEEP and Text2Image scripts named `create_label12_dataset.py` use 100 labels. The skewed generator assigns five, forty, ninety, and one hundred labels across four selectivity segments; base seed is 42 and query seed is 43. Set `QLEAF_DATASET_ROOT` to the root containing its dataset subdirectories and `QLEAF_OUTPUT_ROOT` to the output root. Its YFCC subdirectory is `yfcc10M`.

`--skip-gt` generates labels without GT. The SIFT12 and dataset-specific DEEP/Text2Image100 scripts also calculate GT when run. `uniform100/compute_label_groundtruth.py` computes SIFT GT from `QLEAF_BASE`, `QLEAF_QUERIES`, `QLEAF_BASE_LABELS`, `QLEAF_QUERY_LABELS`, and writes `QLEAF_GT`. The uniform scripts use k=10; the skewed script defaults to k=100 and needs Python FAISS for GT.
