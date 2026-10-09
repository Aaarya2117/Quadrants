import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import modelswap
from modelswap import runtime, weights
from modelswap.arch import MLP, save_checkpoint
from modelswap.real_engine import RealEngine, create_engine
from modelswap.results import EngineError

REGISTRY = """roles:
  classifier:
    architecture: mlp_2layer
    input_dim: 2
    output_dim: 2
    current: "models/model_a.pt"
    previous: "models/model_a.pt"
    candidate: "models/model_b.pt"
    thresholds:
      min_accuracy: 0.85
      max_latency_ms: 5.0
"""


class RealEngineTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.registry = self.dir / "models.yaml"
        self.registry.write_text(REGISTRY, encoding="utf-8")
        save_checkpoint(MLP(input_dim=2, hidden_dim=16, output_dim=2), self.dir / "b.pt")
        self.candidate = str(self.dir / "b.pt")
        self.passing = lambda path: (True, "ok")
        self.failing = lambda path: (False, "boom")
        self.arch_ok = lambda candidate, current, entry: (True, "architecture matches")

    def engine(self, smoke, verify=None):
        return RealEngine(
            self.registry,
            audit_dir=self.dir / "swaps",
            smoke_test=smoke,
            verify=verify or self.arch_ok,
        )

    def test_status_initial(self):
        st = self.engine(self.passing).status("classifier")
        self.assertEqual(st.current, "models/model_a.pt")
        self.assertIsNone(st.previous)

    def test_apply_ok(self):
        result = self.engine(self.passing).apply("classifier", self.candidate)
        self.assertTrue(result.ok)
        self.assertEqual(result.current, self.candidate)
        self.assertEqual(result.previous, "models/model_a.pt")
        self.assertEqual(self.engine(self.passing).status("classifier").current, self.candidate)

    def test_apply_failing_smoke_leaves_file_unchanged(self):
        before = self.registry.read_text(encoding="utf-8")
        result = self.engine(self.failing).apply("classifier", self.candidate)
        self.assertFalse(result.ok)
        self.assertFalse(result.smoke_test_passed)
        self.assertEqual(result.current, "models/model_a.pt")
        self.assertEqual(self.registry.read_text(encoding="utf-8"), before)

    def test_apply_rejects_mismatched_architecture_and_keeps_registry(self):
        before = self.registry.read_text(encoding="utf-8")
        bad_arch = lambda candidate, current, entry: (False, "architecture mismatch: hidden_dim: 16 vs 32")
        result = self.engine(self.passing, verify=bad_arch).apply("classifier", self.candidate)
        self.assertFalse(result.ok)
        self.assertIn("architecture mismatch", result.error)
        self.assertEqual(self.registry.read_text(encoding="utf-8"), before)

    def test_default_architecture_check_rejects_unreadable_candidate(self):
        # The default pre-check reads the candidate's checkpoint; a non-checkpoint must be refused.
        bogus = self.dir / "bogus.pt"
        bogus.write_bytes(b"x")
        eng = RealEngine(self.registry, audit_dir=self.dir / "swaps", smoke_test=self.passing)
        before = self.registry.read_text(encoding="utf-8")
        result = eng.apply("classifier", str(bogus))
        self.assertFalse(result.ok)
        self.assertIn("pre-check failed", result.error)
        self.assertEqual(self.registry.read_text(encoding="utf-8"), before)

    def test_rollback_without_previous_raises(self):
        with self.assertRaises(EngineError):
            self.engine(self.passing).rollback("classifier")

    def test_apply_then_rollback_then_reset(self):
        eng = self.engine(self.passing)
        eng.apply("classifier", self.candidate)
        back = eng.rollback("classifier")
        self.assertTrue(back.ok)
        self.assertEqual(back.current, "models/model_a.pt")
        eng.apply("classifier", self.candidate)
        st = eng.reset("classifier")
        self.assertEqual(st.current, "models/model_a.pt")
        self.assertIsNone(st.previous)

    def test_alias_role(self):
        result = self.engine(self.passing).apply("classifier_role", self.candidate)
        self.assertEqual(result.role, "classifier_role")

    def test_create_engine_defaults(self):
        self.assertEqual(create_engine().registry_path, runtime.DEFAULT_REGISTRY)
        self.assertEqual(create_engine(self.registry).registry_path, self.registry)
        self.assertEqual(create_engine(self.registry, self.dir / "audit").audit_dir, self.dir / "audit")

    def test_apply_reports_weights_and_verifies_conversion(self):
        result = self.engine(self.passing).apply("classifier", self.candidate)
        self.assertTrue(result.ok)
        self.assertIsNotNone(result.weights)
        self.assertIs(result.weights.conversion_verified, True)
        self.assertEqual(result.weights.new.path, self.candidate)
        self.assertEqual(result.weights.new.sha256, weights.describe_pair(None, self.candidate).new.sha256)

    def test_failed_post_write_check_restores_registry_bytes(self):
        before = self.registry.read_bytes()
        unverified = lambda report, active: type(report)(
            previous=report.previous, new=report.new, delta_l2=None, changed_tensors=None,
            conversion_verified=False, note="forced")
        with mock.patch.object(weights, "verify_active", side_effect=unverified):
            result = self.engine(self.passing).apply("classifier", self.candidate)
        self.assertFalse(result.ok)
        self.assertIn("post-write check failed", result.error)
        self.assertEqual(result.current, "models/model_a.pt")
        self.assertEqual(self.registry.read_bytes(), before)

    def test_compare_maps_report(self):
        report = {
            "architecture_ok": True,
            "verdict": "PASS",
            "verdict_reasons": ["candidate accuracy improved"],
            "current": {"accuracy": 0.8, "latency_p95_ms": 1.0},
            "candidate": {"accuracy": 0.9, "latency_p95_ms": 2.0},
        }
        fake = types.ModuleType("modelswap.compare")
        fake.compare = lambda *args, **kwargs: report
        with mock.patch.dict(sys.modules, {"modelswap.compare": fake}), mock.patch.object(
            modelswap, "compare", fake, create=True
        ):
            result = self.engine(self.passing).compare("classifier", self.candidate)
        self.assertEqual(result.model_a.latency_ms, 1.0)
        self.assertEqual(result.model_b.accuracy, 0.9)
        self.assertEqual(result.max_latency_ms, 5.0)
        self.assertTrue(result.eligible)
        self.assertEqual(result.reasons, ("candidate accuracy improved",))

    def test_compare_uses_verdict_from_compare_module(self):
        # The CLI verdict must be Bhagat's verdict, not a recomputed one.
        report = {
            "architecture_ok": True,
            "verdict": "FAIL",
            "verdict_reasons": ["candidate accuracy (80.00%) is below minimum threshold (85.00%)."],
            "current": {"accuracy": 0.8, "latency_p95_ms": 1.0},
            "candidate": {"accuracy": 0.8, "latency_p95_ms": 1.0},
        }
        fake = types.ModuleType("modelswap.compare")
        fake.compare = lambda *args, **kwargs: report
        with mock.patch.dict(sys.modules, {"modelswap.compare": fake}), mock.patch.object(
            modelswap, "compare", fake, create=True
        ):
            result = self.engine(self.passing).compare("classifier", self.candidate)
        self.assertFalse(result.eligible)

    def test_compare_wraps_unexpected_errors(self):
        fake = types.ModuleType("modelswap.compare")
        fake.compare = mock.Mock(side_effect=RuntimeError("corrupt checkpoint"))
        with mock.patch.dict(sys.modules, {"modelswap.compare": fake}), mock.patch.object(
            modelswap, "compare", fake, create=True
        ):
            with self.assertRaises(EngineError) as ctx:
                self.engine(self.passing).compare("classifier", self.candidate)
        self.assertIn("corrupt checkpoint", str(ctx.exception))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_compare_real_models(self):
        repo = Path(__file__).resolve().parent.parent
        eng = RealEngine(repo / "models.yaml", audit_dir=self.dir / "swaps")
        result = eng.compare("classifier", "models/model_b.pt")
        self.assertGreater(result.model_b.accuracy, 0)


if __name__ == "__main__":
    unittest.main()
