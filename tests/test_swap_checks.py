import tempfile
import unittest
from pathlib import Path

import yaml

from modelswap.swap import INITIAL_CURRENT, apply_swap, write_registry_atomic


class SwapCheckOrderTests(unittest.TestCase):
    """Pre-check and smoke test run before models.yaml is written."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.cand = self.dir / "model_b.pt"
        self.cand.write_bytes(b"x")
        self.reg = self.dir / "models.yaml"
        write_registry_atomic(self.reg, {"roles": {"classifier": {
            "current": INITIAL_CURRENT, "previous": INITIAL_CURRENT, "candidate": "models/model_b.pt"}}})
        self.before = self.reg.read_text(encoding="utf-8")

    def test_failed_precheck_skips_smoke_test_and_does_not_write(self):
        calls = []

        def smoke(path):
            calls.append(path)
            return True, "ok"

        result = apply_swap(self.reg, "classifier", str(self.cand), smoke_test=smoke,
                            audit_dir=self.dir / "swaps",
                            verify=lambda c, cur, e: (False, "architecture mismatch"))
        self.assertFalse(result.ok)
        self.assertEqual(calls, [])
        self.assertIn("pre-check failed", result.error)
        self.assertEqual(self.reg.read_text(encoding="utf-8"), self.before)

    def test_precheck_receives_candidate_current_and_entry(self):
        seen = {}

        def verify(candidate, current, entry):
            seen.update(candidate=candidate, current=current, entry=entry)
            return True, "ok"

        apply_swap(self.reg, "classifier", str(self.cand), smoke_test=lambda p: (True, "ok"),
                   audit_dir=self.dir / "swaps", verify=verify)
        self.assertEqual(seen["candidate"], str(self.cand))
        self.assertEqual(seen["current"], INITIAL_CURRENT)
        self.assertEqual(seen["entry"]["candidate"], "models/model_b.pt")

    def test_no_backup_or_temp_files_left_after_failure(self):
        apply_swap(self.reg, "classifier", str(self.cand), smoke_test=lambda p: (False, "bad"),
                   audit_dir=self.dir / "swaps", verify=lambda c, cur, e: (True, "ok"))
        leftovers = [p.name for p in self.dir.iterdir() if p.suffix in (".bak", ".tmp")]
        self.assertEqual(leftovers, [])

    def test_successful_swap_writes_registry(self):
        result = apply_swap(self.reg, "classifier", str(self.cand), smoke_test=lambda p: (True, "ok"),
                            audit_dir=self.dir / "swaps", verify=lambda c, cur, e: (True, "ok"))
        self.assertTrue(result.ok)
        written = yaml.safe_load(self.reg.read_text(encoding="utf-8"))
        self.assertEqual(written["roles"]["classifier"]["current"], str(self.cand))


if __name__ == "__main__":
    unittest.main()
