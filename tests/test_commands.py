import io
import unittest

from modelswap.commands import execute
from modelswap.results import CompareResult, ModelMetrics, RoleStatus, SwapResult


class FakeEngine:
    """Minimal engine double: records calls and returns canned results."""

    def __init__(self, eligible=True, compare_error=None, apply_ok=True):
        self.eligible = eligible
        self.compare_error = compare_error
        self.apply_ok = apply_ok
        self.applied = []

    def compare(self, role, candidate, data=None):
        if self.compare_error is not None:
            raise self.compare_error
        reasons = () if self.eligible else ("candidate accuracy is below minimum",)
        return CompareResult(
            role=role,
            model_a=ModelMetrics("models/model_a.pt", 0.8508, 0.02),
            model_b=ModelMetrics(candidate, 0.9375 if self.eligible else 0.80, 0.014),
            eligible=self.eligible,
            reasons=reasons,
        )

    def apply(self, role, candidate):
        self.applied.append(candidate)
        return SwapResult("apply", role, self.apply_ok, "models/model_a.pt", None, self.apply_ok,
                          None if self.apply_ok else "smoke test failed")

    def rollback(self, role):
        return SwapResult("rollback", role, True, "models/model_a.pt", None, True)

    def status(self, role):
        return RoleStatus(role, "models/model_a.pt", None)

    def reset(self, role):
        return RoleStatus(role, "models/model_a.pt", None)


class ExecuteTests(unittest.TestCase):
    def run_cmd(self, engine, command, **kwargs):
        out = io.StringIO()
        ok = execute(engine, command, role="classifier", out=out, **kwargs)
        return ok, out.getvalue()

    def test_missing_package_is_reported_cleanly(self):
        err = ModuleNotFoundError("No module named 'torch'", name="torch")
        ok, text = self.run_cmd(FakeEngine(compare_error=err), "compare", candidate="models/model_b.pt")
        self.assertFalse(ok)
        self.assertIn("[FAIL] compare: needs a package that is not installed: 'torch'", text)
        self.assertNotIn("Traceback", text)

    def test_apply_refused_when_verdict_fails(self):
        engine = FakeEngine(eligible=False)
        ok, text = self.run_cmd(engine, "apply", candidate="models/model_c.pt")
        self.assertFalse(ok)
        self.assertIn("apply refused", text)
        self.assertEqual(engine.applied, [])

    def test_apply_with_force_skips_verdict(self):
        engine = FakeEngine(eligible=False)
        ok, text = self.run_cmd(engine, "apply", candidate="models/model_c.pt", force=True)
        self.assertTrue(ok)
        self.assertEqual(engine.applied, ["models/model_c.pt"])

    def test_apply_passes_when_verdict_passes(self):
        engine = FakeEngine(eligible=True)
        ok, text = self.run_cmd(engine, "apply", candidate="models/model_b.pt")
        self.assertTrue(ok)
        self.assertIn("[OK] apply", text)

    def test_compare_prints_reasons_and_returns_true(self):
        ok, text = self.run_cmd(FakeEngine(eligible=False), "compare", candidate="models/model_c.pt")
        self.assertTrue(ok)
        self.assertIn("Verdict: FAIL", text)
        self.assertIn("below minimum", text)


if __name__ == "__main__":
    unittest.main()
