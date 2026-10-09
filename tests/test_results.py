import unittest

from modelswap.results import CompareResult, ModelMetrics, eligibility


def _metrics(acc, lat):
    return ModelMetrics("m.pt", acc, lat)


class EligibilityTests(unittest.TestCase):
    def test_better_and_fast_is_eligible(self):
        passed, reasons = eligibility(_metrics(0.8508, 0.02), _metrics(0.9375, 0.014))
        self.assertTrue(passed)
        self.assertEqual(reasons, ())

    def test_equal_accuracy_is_eligible(self):
        passed, _ = eligibility(_metrics(0.90, 0.02), _metrics(0.90, 0.02))
        self.assertTrue(passed)

    def test_accuracy_regression_is_not_eligible(self):
        passed, reasons = eligibility(_metrics(0.94, 0.02), _metrics(0.90, 0.02), min_accuracy=0.85)
        self.assertFalse(passed)
        self.assertTrue(any("regressed" in r for r in reasons))

    def test_below_minimum_accuracy_is_not_eligible(self):
        passed, reasons = eligibility(_metrics(0.80, 0.02), _metrics(0.84, 0.02), min_accuracy=0.85)
        self.assertFalse(passed)
        self.assertTrue(any("minimum" in r for r in reasons))

    def test_latency_over_budget_is_not_eligible(self):
        passed, reasons = eligibility(_metrics(0.82, 0.02), _metrics(0.94, 6.0), max_latency_ms=5.0)
        self.assertFalse(passed)
        self.assertTrue(any("latency" in r for r in reasons))

    def test_latency_exactly_at_budget_is_eligible(self):
        passed, _ = eligibility(_metrics(0.82, 0.02), _metrics(0.94, 5.0), max_latency_ms=5.0)
        self.assertTrue(passed)


class CompareResultTests(unittest.TestCase):
    def test_deltas(self):
        result = CompareResult("r", _metrics(0.82, 2.0), _metrics(0.94, 2.5), eligible=True)
        self.assertAlmostEqual(result.delta_accuracy, 0.12)
        self.assertAlmostEqual(result.delta_latency_ms, 0.5)

    def test_eligible_is_carried_not_recomputed(self):
        # The verdict comes from the engine (Bhagat's verdict()), so it is stored, not derived.
        result = CompareResult("r", _metrics(0.82, 2.0), _metrics(0.94, 2.5), eligible=False, reasons=("x",))
        self.assertFalse(result.eligible)
        self.assertEqual(result.reasons, ("x",))


if __name__ == "__main__":
    unittest.main()
