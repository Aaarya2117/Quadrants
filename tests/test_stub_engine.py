import tempfile
import unittest
from pathlib import Path

from modelswap.results import EngineError
from modelswap.stub_engine import INITIAL_CURRENT, StubEngine

ROLE = "classifier_role"
MODEL_B = "models/model_b.pt"
BROKEN = "models/model_broken.pt"


class StubEngineTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = Path(self._tmp.name) / "state.json"
        self.engine = StubEngine(self.state)

    def test_initial_status_is_model_a(self):
        status = self.engine.status(ROLE)
        self.assertEqual(status.current, INITIAL_CURRENT)
        self.assertIsNone(status.previous)

    def test_state_file_is_not_created_by_status(self):
        self.engine.status(ROLE)
        self.assertFalse(self.state.exists())

    def test_compare_uses_stub_metrics(self):
        result = self.engine.compare(ROLE, MODEL_B)
        self.assertEqual(result.model_a.model, INITIAL_CURRENT)
        self.assertAlmostEqual(result.model_a.accuracy, 0.8508)
        self.assertAlmostEqual(result.model_b.accuracy, 0.9375)
        self.assertTrue(result.eligible)

    def test_apply_promotes_candidate_and_keeps_previous(self):
        result = self.engine.apply(ROLE, MODEL_B)
        self.assertTrue(result.ok)
        self.assertEqual(result.current, MODEL_B)
        self.assertEqual(result.previous, INITIAL_CURRENT)
        self.assertTrue(self.state.exists())
        self.assertEqual(self.engine.status(ROLE).current, MODEL_B)

    def test_rollback_restores_previous(self):
        self.engine.apply(ROLE, MODEL_B)
        result = self.engine.rollback(ROLE)
        self.assertTrue(result.ok)
        self.assertEqual(result.current, INITIAL_CURRENT)
        self.assertEqual(result.previous, MODEL_B)

    def test_rollback_twice_toggles_back(self):
        self.engine.apply(ROLE, MODEL_B)
        self.engine.rollback(ROLE)
        self.assertEqual(self.engine.rollback(ROLE).current, MODEL_B)

    def test_rollback_without_previous_raises(self):
        with self.assertRaises(EngineError):
            self.engine.rollback(ROLE)

    def test_broken_candidate_is_rejected_and_state_unchanged(self):
        result = self.engine.apply(ROLE, BROKEN)
        self.assertFalse(result.ok)
        self.assertFalse(result.smoke_test_passed)
        self.assertIn("smoke test", result.error)
        self.assertEqual(result.current, INITIAL_CURRENT)
        self.assertEqual(self.engine.status(ROLE).current, INITIAL_CURRENT)

    def test_applying_the_active_model_raises(self):
        with self.assertRaises(EngineError):
            self.engine.apply(ROLE, INITIAL_CURRENT)

    def test_reset_restores_initial_state(self):
        self.engine.apply(ROLE, MODEL_B)
        status = self.engine.reset(ROLE)
        self.assertEqual(status.current, INITIAL_CURRENT)
        self.assertIsNone(status.previous)

    def test_roles_are_independent(self):
        self.engine.apply(ROLE, MODEL_B)
        self.assertEqual(self.engine.status("other_role").current, INITIAL_CURRENT)


if __name__ == "__main__":
    unittest.main()
