import unittest

import numpy as np

from ivv_montecarlo_engine import (
    MarketAssumptions,
    SimulationConfig,
    TradingStrategy,
    run_simulation,
)


class MonteCarloEngineTests(unittest.TestCase):
    def setUp(self):
        self.assumptions = MarketAssumptions(initial_price=600.0)
        self.config = SimulationConfig(paths=2_000, days=63, seed=123)
        self.strategy = TradingStrategy()

    def test_results_are_reproducible_and_finite(self):
        first = run_simulation(self.assumptions, self.config, self.strategy)
        second = run_simulation(self.assumptions, self.config, self.strategy)

        np.testing.assert_allclose(first["prices"], second["prices"])
        numeric = first["results"].select_dtypes("number")
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())

    def test_strategy_never_uses_more_than_total_capital(self):
        output = run_simulation(self.assumptions, self.config, self.strategy)
        results = output["results"]

        self.assertLessEqual(results["capital_used"].max(), 1.0)
        self.assertGreaterEqual((1 + results["strategy_return_usd"]).min(), 0.0)

    def test_scenario_mix_is_close_to_configured_probabilities(self):
        config = SimulationConfig(paths=20_000, days=10, seed=456)
        output = run_simulation(self.assumptions, config, self.strategy)
        observed = (
            np.bincount(output["scenario_ids"], minlength=3) / config.paths
        )

        np.testing.assert_allclose(observed, [0.15, 0.70, 0.15], atol=0.015)

    def test_percentile_order_is_monotonic(self):
        output = run_simulation(self.assumptions, self.config, self.strategy)
        values = output["percentiles"]["strategy_return_usd"].to_numpy()

        self.assertTrue(np.all(np.diff(values) >= 0))

if __name__ == "__main__":
    unittest.main()
