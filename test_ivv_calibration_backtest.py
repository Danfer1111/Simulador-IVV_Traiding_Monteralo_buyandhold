import unittest

import numpy as np
import pandas as pd

from ivv_calibration_backtest import (
    BacktestConfig,
    calibrate_history,
    run_walk_forward_backtest,
)
from ivv_montecarlo_engine import TradingStrategy


def synthetic_history(rows=1_200, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=rows)
    common = rng.standard_t(6, rows) * 0.008
    oil_returns = rng.normal(0.0001, 0.018, rows)
    fx_returns = -0.25 * common + rng.normal(0, 0.005, rows)
    rate_changes = rng.normal(0, 0.015, rows)
    ivv_returns = (
        0.0003
        - 0.08 * oil_returns
        - 0.002 * rate_changes
        + common
    )
    return pd.DataFrame(
        {
            "ivv": 250 * np.exp(np.cumsum(ivv_returns)),
            "oil": 60 * np.exp(np.cumsum(oil_returns)),
            "rate_10y": 3.0 + np.cumsum(rate_changes),
            "usdmxn": 19 * np.exp(np.cumsum(fx_returns)),
        },
        index=dates,
    )


class CalibrationBacktestTests(unittest.TestCase):
    def test_calibration_is_finite_and_preserves_scenario_weights(self):
        result = calibrate_history(synthetic_history())

        self.assertGreater(result.observations, 500)
        self.assertGreater(result.student_df, 2)
        self.assertTrue(
            np.isclose(sum(item.probability for item in result.scenarios), 1.0)
        )
        self.assertTrue(
            np.isfinite(np.array(list(result.diagnostics.values()))).all()
        )

    def test_walk_forward_uses_requested_number_of_windows(self):
        results, summary = run_walk_forward_backtest(
            synthetic_history(),
            TradingStrategy(),
            BacktestConfig(
                training_days=504,
                horizon_days=63,
                step_days=126,
                paths=1_000,
                max_windows=4,
            ),
        )

        self.assertEqual(len(results), 4)
        self.assertEqual(summary["windows"], 4)
        self.assertTrue(results["price_covered"].dtype == bool)
        self.assertTrue(
            np.isfinite(results.select_dtypes("number").to_numpy()).all()
        )


if __name__ == "__main__":
    unittest.main()
