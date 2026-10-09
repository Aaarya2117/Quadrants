from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from modelswap import runtime
from modelswap.results import DEFAULT_MAX_LATENCY_MS, EngineError

YAML = """roles:
  classifier:
    architecture: mlp_2layer
    current: "models/model_b.pt"
    previous: "models/model_a.pt"
    thresholds:
      max_latency_ms: 5.0
"""


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "models.yaml"
        self.path.write_text(YAML)

    def test_default_registry_is_models_yaml(self):
        self.assertEqual(runtime.DEFAULT_REGISTRY, Path("models.yaml"))

    def test_load_registry_errors(self):
        with self.assertRaises(EngineError):
            runtime.load_registry(Path(self.tmp.name) / "nope.yaml")
        empty = Path(self.tmp.name) / "e.yaml"
        empty.write_text("")
        with self.assertRaises(EngineError):
            runtime.load_registry(empty)
        empty.write_text("foo: 1\n")
        with self.assertRaises(EngineError):
            runtime.load_registry(empty)

    def test_resolve_role(self):
        config = runtime.load_registry(self.path)
        self.assertEqual(runtime.resolve_role(config, "classifier"), "classifier")
        self.assertEqual(runtime.resolve_role(config, "classifier_role"), "classifier")
        with self.assertRaises(EngineError) as ctx:
            runtime.resolve_role(config, "bogus")
        self.assertIn("classifier", str(ctx.exception))

    def test_status_alias_and_previous(self):
        status = runtime.get_status(self.path, "classifier_role")
        self.assertEqual(status.role, "classifier_role")
        self.assertEqual(status.current, "models/model_b.pt")
        self.assertEqual(status.previous, "models/model_a.pt")

    def test_status_previous_equal_current_is_none(self):
        self.path.write_text(YAML.replace("model_b.pt", "model_a.pt"))
        self.assertIsNone(runtime.get_status(self.path, "classifier").previous)

    def test_max_latency(self):
        config = runtime.load_registry(self.path)
        self.assertEqual(runtime.get_max_latency(config, "classifier"), 5.0)
        del config["roles"]["classifier"]["thresholds"]
        self.assertEqual(runtime.get_max_latency(config, "classifier"), DEFAULT_MAX_LATENCY_MS)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
    def test_get_loads_eval_model(self):
        model = runtime.get("classifier")
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
