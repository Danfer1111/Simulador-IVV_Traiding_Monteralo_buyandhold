import unittest

import numpy as np

from ivv_montecarlo_engine import (
    AggressiveTradingStrategy,
    MarketAssumptions,
    SimulationConfig,
    TradingStrategy,
    run_aggressive_trading,
    run_simulation,
    summarize_aggressive_trading,
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

    def test_aggressive_immediate_entry_tracks_future_maximum(self):
        prices = np.array(
            [
                [100.0, 96.0, 104.0, 110.0, 107.0],
                [100.0, 102.0, 99.0, 97.0, 95.0],
            ]
        )
        strategy = AggressiveTradingStrategy(
            capital_usd=1_000,
            entry_mode="immediate",
            take_profit=0.50,
            stop_loss=0.50,
            trailing_stop=0.50,
            max_trades=1,
            transaction_cost_bps=0,
        )
        results = run_aggressive_trading(prices, strategy)

        self.assertTrue((results["first_entry_day"] == 0).all())
        self.assertEqual(results.loc[0, "best_day_after_entry"], 3)
        self.assertAlmostEqual(results.loc[0, "theoretical_max_value"], 1_100)

    def test_aggressive_buy_dip_can_remain_in_cash(self):
        prices = np.array([[100.0, 101.0, 102.0, 103.0]])
        results = run_aggressive_trading(
            prices,
            AggressiveTradingStrategy(
                entry_mode="buy_dip",
                entry_drawdown=0.05,
            ),
        )
        summary = summarize_aggressive_trading(results)

        self.assertFalse(results.loc[0, "traded"])
        self.assertEqual(results.loc[0, "final_capital"], 1_000)
        self.assertEqual(summary["probability_trade"], 0.0)


if __name__ == "__main__":
    unittest.main()
