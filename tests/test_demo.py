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

    def test_interactive_menu_exit(self):
        from unittest import mock
        from modelswap.demo import run_interactive_menu
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=["0"]):
            code = run_interactive_menu(self.engine, "stub", out=out)
        self.assertEqual(code, 0)
        self.assertIn("Exiting ML Model Swap demo", out.getvalue())

    def test_interactive_menu_runs_status(self):
        from unittest import mock
        from modelswap.demo import run_interactive_menu
        out = io.StringIO()
        with mock.patch("builtins.input", side_effect=["1", "1", "", "0"]):
            code = run_interactive_menu(self.engine, "stub", out=out)
        self.assertEqual(code, 0)
        self.assertIn("Running: status", out.getvalue())


if __name__ == "__main__":
    unittest.main()
