import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import modelswap
from modelswap import runtime
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
        (self.dir / "b.pt").write_bytes(b"x")
        self.candidate = str(self.dir / "b.pt")
        self.passing = lambda path: (True, "ok")
        self.failing = lambda path: (False, "boom")

    def engine(self, smoke):
        return RealEngine(self.registry, audit_dir=self.dir / "swaps", smoke_test=smoke)

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

    def test_create_engine_path_rule(self):
        self.assertEqual(create_engine(None).registry_path, runtime.DEFAULT_REGISTRY)
        self.assertEqual(create_engine(runtime.CLI_DEFAULT_STATE).registry_path, runtime.DEFAULT_REGISTRY)
        self.assertEqual(create_engine(self.registry).registry_path, self.registry)

    def test_compare_maps_report(self):
        report = {
            "architecture_ok": True,
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_compare_real_models(self):
        repo = Path(__file__).resolve().parent.parent
        eng = RealEngine(repo / "models.yaml", audit_dir=self.dir / "swaps")
        result = eng.compare("classifier", "models/model_b.pt")
        self.assertGreater(result.model_b.accuracy, 0)


if __name__ == "__main__":
    unittest.main()
