import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILDER = os.environ.get(
    "QLEAF_BUILDER", os.path.join(ROOT, "build", "bin", "build_nsg_subgraphs_ml")
)

class OverwriteGuardTest(unittest.TestCase):
    def test_builder_refuses_existing_graph_before_loading_inputs(self):
        if not os.path.exists(BUILDER):
            self.skipTest(f"build_nsg_subgraphs_ml not built: {BUILDER}")
        with tempfile.TemporaryDirectory() as tmp:
            graph_dir = os.path.join(tmp, "out", "tiny", "auto_tiers")
            os.makedirs(graph_dir)
            with open(os.path.join(graph_dir, "t1_nsg.graph"), "wb"):
                pass
            result = subprocess.run(
                [BUILDER, "missing.fvecs", "missing.tsv", "missing-idx", os.path.join(tmp, "out"), "tiny"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing to overwrite", result.stdout)

            env = os.environ.copy()
            env["QLEAF_OVERWRITE"] = "1"
            allowed = subprocess.run(
                [BUILDER, "missing.fvecs", "missing.tsv", "missing-idx", os.path.join(tmp, "out"), "tiny"],
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(allowed.returncode, 0)
            self.assertNotIn("refusing to overwrite", allowed.stdout)

    def test_yfcc_export_refuses_nonempty_directory_before_loading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "export")
            os.makedirs(out)
            with open(os.path.join(out, "sentinel"), "w") as f:
                f.write("keep")
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, "data_prepare", "yfcc_prepare.py"),
                    "export",
                    "--base-meta",
                    "missing-base.spmat",
                    "--query-meta",
                    "missing-query.spmat",
                    "--query-u8bin",
                    "missing-query.u8bin",
                    "--gt",
                    "missing.ibin",
                    "--out",
                    out,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output directory is not empty", result.stdout)
            self.assertTrue(os.path.exists(os.path.join(out, "sentinel")))

            allowed = subprocess.run(
                [
                    sys.executable,
                    os.path.join(ROOT, "data_prepare", "yfcc_prepare.py"),
                    "export",
                    "--base-meta",
                    "missing-base.spmat",
                    "--query-meta",
                    "missing-query.spmat",
                    "--query-u8bin",
                    "missing-query.u8bin",
                    "--gt",
                    "missing.ibin",
                    "--out",
                    out,
                    "--overwrite",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertNotEqual(allowed.returncode, 0)
            self.assertNotIn("output directory is not empty", allowed.stdout)

if __name__ == "__main__":
    unittest.main()
