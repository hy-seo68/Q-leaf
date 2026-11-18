import csv
import os
import struct
import subprocess
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.environ.get("QLEAF_BENCH", os.path.join(ROOT, "build", "bin", "qleaf_bench"))

def write_vecs(path, rows, kind="f"):
    with open(path, "wb") as f:
        for row in rows:
            f.write(struct.pack("i", len(row)))
            f.write(struct.pack(f"{len(row)}{kind}", *row))

class QleafBenchOutputTest(unittest.TestCase):
    def test_per_query_and_percentile_columns(self):
        if not os.path.exists(BINARY):
            self.skipTest(f"qleaf_bench not built: {BINARY}")
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "base.fvecs")
            query = os.path.join(tmp, "query.fvecs")
            gt = os.path.join(tmp, "gt.ivecs")
            graph = os.path.join(tmp, "a_nsg.graph")
            idx = os.path.join(tmp, "a.idx")
            entry = os.path.join(tmp, "a_entry_points.txt")
            labels = os.path.join(tmp, "labels.txt")
            config = os.path.join(tmp, "config.csv")
            aggregate = os.path.join(tmp, "aggregate.tsv")
            per_query = os.path.join(tmp, "per_query.tsv")
            result_ids = os.path.join(tmp, "result_ids.tsv")

            write_vecs(base, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 2.0)])
            write_vecs(query, [(0.1, 0.0), (1.9, 2.0)])
            write_vecs(gt, [(0, 1), (3, 1)], "i")
            with open(idx, "wb") as f:
                f.write(struct.pack("5I", 4, 0, 1, 2, 3))
            with open(graph, "wb") as f:
                words = [2, 0, 2, 1, 2, 2, 0, 3, 2, 0, 3, 2, 1, 2]
                f.write(struct.pack(f"{len(words)}I", *words))
            with open(entry, "w") as f:
                f.write("0 | 0\n")
                f.write("0 | 0\n")
            with open(labels, "w") as f:
                f.write("a\na\n")
            with open(config, "w") as f:
                f.write(f"a,{graph},{idx},{entry}\n")

            command = [
                BINARY, "--base", base, "--config", config, "--queries", query, "--labels", labels,
                "--label-format", "plain", "--gt", gt, "--k", "2", "--L", "2,3", "--init", "nsg",
                "--layout", "packed", "--runs", "2", "--warmup", "1", "--out", aggregate,
                "--per-query-out", per_query, "--result-out", result_ids,
            ]
            first = subprocess.run(
                command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self.assertIn(f"entry points {entry}: original=2, unique=1", first.stdout)

            with open(aggregate, newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(rows), 4)
            self.assertIn("ndc_search_p50", rows[0])
            self.assertIn("ndc_search_p99", rows[0])
            self.assertIn("latency_us_p50", rows[0])
            self.assertIn("latency_us_p99", rows[0])

            with open(per_query, newline="") as f:
                per_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(per_rows), 8)
            self.assertEqual(set(per_rows[0]),
                             {"run", "L", "qid", "recall_i", "ndc_entry", "ndc_search", "latency_us"})

            with open(result_ids, newline="") as f:
                result_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(result_rows), 16)
            self.assertEqual(set(result_rows[0]), {"run", "L", "qid", "rank", "global_id"})

            bitset_aggregate = os.path.join(tmp, "bitset_aggregate.tsv")
            bitset_per_query = os.path.join(tmp, "bitset_per_query.tsv")
            bitset_result_ids = os.path.join(tmp, "bitset_result_ids.tsv")
            bitset_command = list(command)
            bitset_command[bitset_command.index(aggregate)] = bitset_aggregate
            bitset_command[bitset_command.index(per_query)] = bitset_per_query
            bitset_command[bitset_command.index(result_ids)] = bitset_result_ids
            bitset_command += ["--visited", "bitset"]
            bitset_run = subprocess.run(
                bitset_command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self.assertIn("visited: bitset", bitset_run.stdout)
            with open(bitset_result_ids, newline="") as f:
                bitset_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(bitset_rows, result_rows)

            optimized_aggregate = os.path.join(tmp, "optimized_aggregate.tsv")
            optimized_per_query = os.path.join(tmp, "optimized_per_query.tsv")
            optimized_result_ids = os.path.join(tmp, "optimized_result_ids.tsv")
            optimized_command = list(command)
            optimized_command[optimized_command.index(aggregate)] = optimized_aggregate
            optimized_command[optimized_command.index(per_query)] = optimized_per_query
            optimized_command[optimized_command.index(result_ids)] = optimized_result_ids
            optimized_command += ["--packed-record", "optimized"]
            optimized_run = subprocess.run(
                optimized_command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self.assertIn("packed record: optimized", optimized_run.stdout)
            with open(optimized_result_ids, newline="") as f:
                optimized_rows = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(optimized_rows, result_rows)

            refused = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("output already exists", refused.stdout)

            subprocess.run(command + ["--overwrite"], check=True, stdout=subprocess.DEVNULL)
            with open(per_query, newline="") as f:
                self.assertEqual(len(list(csv.DictReader(f, delimiter="\t"))), 8)

            missing_dir = os.path.join(tmp, "missing")
            bad_command = list(command)
            bad_command[bad_command.index(aggregate)] = os.path.join(missing_dir, "aggregate.tsv")
            bad_command[bad_command.index(per_query)] = os.path.join(missing_dir, "per_query.tsv")
            bad_command[bad_command.index(result_ids)] = os.path.join(missing_dir, "result_ids.tsv")
            failed_open = subprocess.run(
                bad_command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
            self.assertNotEqual(failed_open.returncode, 0)
            self.assertIn("cannot create output", failed_open.stdout)

if __name__ == "__main__":
    unittest.main()
