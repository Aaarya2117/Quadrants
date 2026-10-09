import json
import tempfile
import unittest
from pathlib import Path

import yaml

from modelswap.results import EngineError
from modelswap.swap import INITIAL_CURRENT, apply_swap, reset_registry, rollback_swap, write_registry_atomic


def ok(_path):
    return True, "ok"


def bad(_path):
    return False, "forward pass failed"


class SwapTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.audit = self.dir / "swaps"
        self.cand = self.dir / "model_b.pt"
        self.cand.write_bytes(b"x")
        self.cand_s = str(self.cand)
        self.reg = self.dir / "models.yaml"
        self.cfg = {"roles": {"classifier": {
            "architecture": "mlp_2layer", "current": INITIAL_CURRENT, "previous": INITIAL_CURRENT,
            "candidate": "models/model_b.pt", "thresholds": {"max_latency_ms": 5.0}}}}
        write_registry_atomic(self.reg, self.cfg)

    def roles(self):
        return yaml.safe_load(self.reg.read_text())["roles"]["classifier"]

    def test_apply_ok_and_audit(self):
        r = apply_swap(self.reg, "classifier_role", self.cand_s, smoke_test=ok, audit_dir=self.audit)
        self.assertTrue(r.ok)
        self.assertEqual((r.role, r.current, r.previous), ("classifier_role", self.cand_s, INITIAL_CURRENT))
        self.assertEqual(self.roles()["current"], self.cand_s)
        self.assertEqual(self.roles()["candidate"], "models/model_b.pt")
        files = list(self.audit.glob("*-apply-classifier_role.json"))
        self.assertEqual(len(files), 1)
        rec = json.loads(files[0].read_text())
        for k in ("operation", "role", "ok", "from_model", "to_model", "smoke_test_passed",
                  "error", "timestamp_utc", "duration_ms"):
            self.assertIn(k, rec)
        self.assertEqual(rec["to_model"], self.cand_s)

    def test_failure_restores_file(self):
        before = self.reg.read_text()
        r = apply_swap(self.reg, "classifier", self.cand_s, smoke_test=bad, audit_dir=self.audit)
        self.assertFalse(r.ok)
        self.assertFalse(r.smoke_test_passed)
        self.assertEqual(r.error, "forward pass failed")
        self.assertEqual(r.current, INITIAL_CURRENT)
        self.assertEqual(self.reg.read_text(), before)
        rec = json.loads(next(self.audit.glob("*.json")).read_text())
        self.assertFalse(rec["ok"])

    def test_apply_errors(self):
        with self.assertRaises(EngineError):
            apply_swap(self.reg, "nope", self.cand_s, smoke_test=ok, audit_dir=self.audit)
        with self.assertRaises(EngineError):
            apply_swap(self.reg, "classifier", INITIAL_CURRENT, smoke_test=ok, audit_dir=self.audit)
        with self.assertRaises(EngineError):
            apply_swap(self.reg, "classifier", str(self.dir / "missing.pt"), smoke_test=ok, audit_dir=self.audit)

    def test_rollback_without_previous(self):
        with self.assertRaises(EngineError) as cm:
            rollback_swap(self.reg, "classifier", smoke_test=ok, audit_dir=self.audit)
        self.assertIn("nothing to roll back to", str(cm.exception))

    def test_rollback_and_failure(self):
        apply_swap(self.reg, "classifier", self.cand_s, smoke_test=ok, audit_dir=self.audit)
        before = self.reg.read_text()
        r = rollback_swap(self.reg, "classifier", smoke_test=bad, audit_dir=self.audit)
        self.assertFalse(r.ok)
        self.assertEqual(r.current, self.cand_s)
        self.assertEqual(self.reg.read_text(), before)
        r = rollback_swap(self.reg, "classifier", smoke_test=ok, audit_dir=self.audit)
        self.assertTrue(r.ok)
        self.assertEqual((r.current, r.previous), (INITIAL_CURRENT, self.cand_s))
        self.assertEqual(len(list(self.audit.glob("*-rollback-*.json"))), 2)

    def test_reset(self):
        apply_swap(self.reg, "classifier", self.cand_s, smoke_test=ok, audit_dir=self.audit)
        s = reset_registry(self.reg, "classifier")
        self.assertEqual((s.current, s.previous), (INITIAL_CURRENT, None))
        self.assertEqual(self.roles()["previous"], INITIAL_CURRENT)

    def test_no_leftover_temp_files(self):
        apply_swap(self.reg, "classifier", self.cand_s, smoke_test=ok, audit_dir=self.audit)
        rollback_swap(self.reg, "classifier", smoke_test=ok, audit_dir=self.audit)
        reset_registry(self.reg, "classifier")
        self.assertFalse(list(self.dir.glob("*.tmp")) + list(self.dir.glob(".*.tmp")))


if __name__ == "__main__":
    unittest.main()
