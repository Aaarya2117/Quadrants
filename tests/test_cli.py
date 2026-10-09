import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modelswap.cli import main
from modelswap.engine import REAL_MODULE

CANDIDATE = "models/model_b.pt"


class CliTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.state = str(Path(self._tmp.name) / "state.json")

    def run_cli(self, *args):
        """Run the CLI against the stub with a temp state file. Returns (exit_code, stdout)."""
        out = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["--backend", "stub", "--state-file", self.state, *args], out=out)
        return code, out.getvalue()

    def test_status_shows_model_a_and_no_previous(self):
        code, text = self.run_cli("status")
        self.assertEqual(code, 0)
        self.assertIn("Current:  models/model_a.pt", text)
        self.assertIn("Previous: -", text)

    def test_compare_reports_pass_verdict(self):
        code, text = self.run_cli("compare", "--candidate", CANDIDATE)
        self.assertEqual(code, 0)
        self.assertIn("Verdict: PASS", text)
        self.assertIn("+8.7 pts", text)

    def test_apply_then_status_shows_candidate(self):
        code, text = self.run_cli("apply", "--candidate", CANDIDATE)
        self.assertEqual(code, 0)
        self.assertIn("[OK] apply", text)
        self.assertIn("Smoke test: PASS", text)
        _, status_text = self.run_cli("status")
        self.assertIn(f"Current:  {CANDIDATE}", status_text)

    def test_apply_rollback_round_trip(self):
        self.run_cli("apply", "--candidate", CANDIDATE)
        code, text = self.run_cli("rollback")
        self.assertEqual(code, 0)
        self.assertIn("[OK] rollback", text)
        _, status_text = self.run_cli("status")
        self.assertIn("Current:  models/model_a.pt", status_text)

    def test_rollback_without_previous_fails(self):
        code, text = self.run_cli("rollback")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] rollback", text)
        self.assertIn("nothing to roll back to", text)

    def test_broken_candidate_is_refused_by_verdict_without_force(self):
        code, text = self.run_cli("apply", "--candidate", "models/model_broken.pt")
        self.assertEqual(code, 1)
        self.assertIn("apply refused", text)
        _, status_text = self.run_cli("status")
        self.assertIn("Current:  models/model_a.pt", status_text)

    def test_broken_candidate_fails_smoke_test_with_force(self):
        code, text = self.run_cli("apply", "--candidate", "models/model_broken.pt", "--force")
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] apply rejected", text)
        self.assertIn("Active model unchanged: models/model_a.pt", text)

    def test_candidate_failing_verdict_is_refused_unless_forced(self):
        code, text = self.run_cli("apply", "--candidate", "models/model_c.pt")
        self.assertEqual(code, 1)
        self.assertIn("apply refused", text)
        code, text = self.run_cli("apply", "--candidate", "models/model_c.pt", "--force")
        self.assertEqual(code, 0)
        self.assertIn("[OK] apply", text)

    def test_apply_same_model_fails(self):
        code, text = self.run_cli("apply", "--candidate", "models/model_a.pt")
        self.assertEqual(code, 1)
        self.assertIn("already the active model", text)

    def test_reset_restores_model_a(self):
        self.run_cli("apply", "--candidate", CANDIDATE)
        code, text = self.run_cli("reset")
        self.assertEqual(code, 0)
        self.assertIn("Current:  models/model_a.pt", text)

    def test_apply_requires_candidate(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--backend", "stub", "--state-file", self.state, "apply"], out=io.StringIO())
        self.assertEqual(ctx.exception.code, 2)

    def test_real_backend_missing_reports_failure(self):
        out = io.StringIO()
        with mock.patch.dict(sys.modules, {REAL_MODULE: None}):
            code = main(["--backend", "real", "--state-file", self.state, "status"], out=out)
        self.assertEqual(code, 1)
        self.assertIn("does not exist yet", out.getvalue())


if __name__ == "__main__":
    unittest.main()
