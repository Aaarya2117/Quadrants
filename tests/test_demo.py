import io
import tempfile
import unittest
from pathlib import Path

from modelswap.demo import DemoOptions, build_steps, run_demo
from modelswap.results import SwapResult
from modelswap.stub_engine import INITIAL_CURRENT, StubEngine

ROLE = "classifier_role"


class DemoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.engine = StubEngine(Path(self._tmp.name) / "state.json")

    def run_demo_quiet(self, **kwargs):
        out = io.StringIO()
        code = run_demo(
            self.engine,
            "stub",
            DemoOptions(fast=True, **kwargs),
            out=out,
            pause=lambda message: None,
        )
        return code, out.getvalue()

    def test_default_demo_passes_and_ends_on_model_a(self):
        code, text = self.run_demo_quiet()
        self.assertEqual(code, 0)
        self.assertIn("Result: PASS (6/6 steps)", text)
        self.assertEqual(self.engine.status(ROLE).current, INITIAL_CURRENT)

    def test_demo_with_failure_step_passes_and_shows_rejection(self):
        code, text = self.run_demo_quiet(show_failure=True)
        self.assertEqual(code, 0)
        self.assertIn("expected rejection", text)
        self.assertIn("Result: PASS (7/7 steps)", text)

    def test_demo_prints_stub_warning(self):
        _, text = self.run_demo_quiet()
        self.assertIn("Backend is the stub", text)

    def test_demo_pauses_between_steps_when_not_fast(self):
        pauses = []
        run_demo(
            self.engine,
            "stub",
            DemoOptions(fast=False),
            out=io.StringIO(),
            pause=pauses.append,
        )
        self.assertEqual(len(pauses), len(build_steps(DemoOptions())) - 1)

    def test_failed_expectation_stops_demo(self):
        # Force a mismatch: the swap step expects success, so a failing engine result must fail the run.
        self.engine.apply = lambda role, candidate: SwapResult(
            "apply", role, False, INITIAL_CURRENT, None, False, error="forced"
        )
        code, text = self.run_demo_quiet()
        self.assertEqual(code, 1)
        self.assertIn("did not behave as expected", text)


if __name__ == "__main__":
    unittest.main()
