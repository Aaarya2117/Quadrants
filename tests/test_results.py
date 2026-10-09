import unittest

from modelswap.results import CompareResult, ModelMetrics


def _compare(acc_a, acc_b, lat_b, lat_a=2.0, max_latency=15.0):
    return CompareResult(
        role="r",
        model_a=ModelMetrics("models/model_a.pt", acc_a, lat_a),
        model_b=ModelMetrics("models/model_b.pt", acc_b, lat_b),
        max_latency_ms=max_latency,
    )


class CompareEligibilityTests(unittest.TestCase):
    def test_better_and_fast_is_eligible(self):
        self.assertTrue(_compare(0.82, 0.94, 2.1).eligible)

    def test_equal_accuracy_is_eligible(self):
        self.assertTrue(_compare(0.90, 0.90, 2.0).eligible)

    def test_accuracy_drop_is_not_eligible(self):
        self.assertFalse(_compare(0.94, 0.82, 2.0).eligible)

    def test_latency_over_budget_is_not_eligible(self):
        self.assertFalse(_compare(0.82, 0.94, 16.0).eligible)

    def test_latency_exactly_at_budget_is_eligible(self):
        self.assertTrue(_compare(0.82, 0.94, 15.0).eligible)

    def test_deltas(self):
        result = _compare(0.82, 0.94, 2.5, lat_a=2.0)
        self.assertAlmostEqual(result.delta_accuracy, 0.12)
        self.assertAlmostEqual(result.delta_latency_ms, 0.5)


if __name__ == "__main__":
    unittest.main()
