import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BUILDER = Path(os.environ.get("QLEAF_BUILDER", ROOT / "build/bin/build_nsg_subgraphs_ml"))

class HugeScheduleTest(unittest.TestCase):

    TAG_SIZE = 128

    def run_builder(self, minimum, large_threads=4):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            idx = root / "idx"
            idx.mkdir()
            rng = np.random.default_rng(42)
            vectors = np.empty((self.TAG_SIZE * 2, 17), dtype=np.float32)
            vectors.view(np.int32)[:, 0] = 16
            vectors[:, 1:] = rng.normal(size=(self.TAG_SIZE * 2, 16))
            vectors.tofile(root / "base.fvecs")
            for name, start in (("t0", 0), ("t1", self.TAG_SIZE)):
                np.r_[np.uint32(self.TAG_SIZE), np.arange(start, start + self.TAG_SIZE, dtype=np.uint32)].astype(np.uint32).tofile(idx / (name + ".idx"))
            (root / "tags.tsv").write_text(f"#name\ttag\tsize\nt0\t0\t{self.TAG_SIZE}\nt1\t1\t{self.TAG_SIZE}\n")
            env = dict(os.environ, QLEAF_BUILD_THREADS="2", QLEAF_LARGE_THREADS=str(large_threads), QLEAF_OVERWRITE="0")
            env.pop("QLEAF_HUGE_MIN_SIZE", None)
            if minimum is not None:
                env["QLEAF_HUGE_MIN_SIZE"] = minimum
            result = subprocess.run(
                [str(BUILDER), str(root / "base.fvecs"), str(root / "tags.tsv"), str(idx), str(root / "out"), "tiny"],
                env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
            )
            report = root / "out/tiny/auto_tiers/build_tasks.tsv"
            rows = []
            if report.exists():
                rows = list(csv.DictReader((line for line in report.read_text().splitlines() if not line.startswith("#")), delimiter="\t"))
            return result, rows

    def test_default_keeps_strict_above_total_div_threads(self):
        result, rows = self.run_builder(None)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("  huge ", result.stdout)
        self.assertEqual([r["omp_threads"] for r in rows], ["1", "1"])

    def test_explicit_minimum_is_inclusive_and_runs_first_sequentially(self):
        result, rows = self.run_builder(str(self.TAG_SIZE))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([r["omp_threads"] for r in rows], ["4", "4"])
        self.assertTrue(all(r["knn_team"] == "4" and r["nsg_team"] == "4" for r in rows))
        self.assertLess(result.stdout.index("huge t0"), result.stdout.index("[t1] KNN start"))

    def test_explicit_minimum_still_runs_first_when_threads_is_one(self):
        result, rows = self.run_builder(str(self.TAG_SIZE), large_threads=1)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("  huge "), 2)
        self.assertEqual([r["omp_threads"] for r in rows], ["1", "1"])

    def test_invalid_minimum_is_rejected(self):
        result, rows = self.run_builder("-1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid QLEAF_HUGE_MIN_SIZE", result.stdout)
        self.assertEqual(rows, [])

if __name__ == "__main__":
    unittest.main()
