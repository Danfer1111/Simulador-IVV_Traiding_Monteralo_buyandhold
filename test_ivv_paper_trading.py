import unittest

import numpy as np

from ivv_paper_trading import (
    PaperTradingConfig,
    PaperTradingStrategy,
    run_paper_trading,
)


class PaperTradingTests(unittest.TestCase):
    def setUp(self):
        self.strategy = PaperTradingStrategy(
            drawdown_levels=(0.03,),
            allocations=(0.50,),
            take_profit=0.20,
            stop_loss=0.05,
            trailing_stop=0.10,
        )

    def test_limit_order_is_filled_on_a_later_session(self):
        output = run_paper_trading(
            np.array([100.0, 96.0, 95.0, 97.0]),
            self.strategy,
            PaperTradingConfig(
                spread_bps=0,
                slippage_bps=0,
                transaction_cost_bps=0,
                limit_offset_bps=0,
            ),
        )

        events = output["events"]
        self.assertTrue((events["event"] == "Orden creada").any())
        self.assertTrue(
            (
                (events["event"] == "Orden ejecutada")
                & (events["side"] == "Compra")
            ).any()
        )
        self.assertGreater(output["sessions"].iloc[-1]["position_shares"], 0)

    def test_limit_order_can_expire_without_a_fill(self):
        output = run_paper_trading(
            np.array([100.0, 96.0, 96.5, 97.0]),
            self.strategy,
            PaperTradingConfig(
                spread_bps=0,
                slippage_bps=0,
                transaction_cost_bps=0,
                limit_offset_bps=200,
                intraday_range_bps=0,
                order_expiry_sessions=1,
            ),
        )

        self.assertEqual(output["summary"]["orders_filled"], 0)
        self.assertGreaterEqual(output["summary"]["orders_expired"], 1)

    def test_exposure_limit_rejects_oversized_order(self):
        output = run_paper_trading(
            np.array([100.0, 96.0, 95.0]),
            self.strategy,
            PaperTradingConfig(max_exposure=0.10),
        )

        self.assertEqual(output["summary"]["orders_filled"], 0)
        self.assertEqual(output["summary"]["orders_rejected"], 1)
        self.assertIn(
            "Supera la exposicion maxima",
            output["events"]["reason"].tolist(),
        )

    def test_stop_loss_closes_a_filled_position(self):
        output = run_paper_trading(
            np.array([100.0, 96.0, 95.0, 85.0]),
            self.strategy,
            PaperTradingConfig(
                spread_bps=0,
                slippage_bps=0,
                transaction_cost_bps=0,
                limit_offset_bps=0,
            ),
        )

        sales = output["events"].loc[
            (output["events"]["side"] == "Venta")
            & (output["events"]["status"] == "Ejecutada")
        ]
        self.assertIn("Stop-loss", sales["reason"].tolist())
        self.assertFalse(output["summary"]["open_position"])

    def test_portfolio_loss_limit_halts_new_trading(self):
        output = run_paper_trading(
            np.array([100.0, 96.0, 95.0, 80.0, 78.0]),
            PaperTradingStrategy(
                drawdown_levels=(0.03,),
                allocations=(1.0,),
                take_profit=0.50,
                stop_loss=0.50,
                trailing_stop=0.50,
            ),
            PaperTradingConfig(
                spread_bps=0,
                slippage_bps=0,
                transaction_cost_bps=0,
                limit_offset_bps=0,
                max_exposure=1.0,
                max_portfolio_loss=0.05,
            ),
        )

        self.assertTrue(output["summary"]["risk_halted"])
        self.assertIn(
            "Sistema detenido",
            output["events"]["event"].tolist(),
        )
        self.assertFalse(output["summary"]["open_position"])

    def test_theoretical_trade_buys_low_and_sells_at_a_later_high(self):
        output = run_paper_trading(
            np.array([100.0, 90.0, 95.0, 120.0]),
            self.strategy,
            PaperTradingConfig(
                spread_bps=0,
                slippage_bps=0,
                transaction_cost_bps=0,
                intraday_range_bps=0,
                max_exposure=1.0,
            ),
        )

        theoretical = output["theoretical"]
        self.assertTrue(theoretical["trade_available"])
        self.assertEqual(theoretical["entry_session"], 1)
        self.assertEqual(theoretical["exit_session"], 3)
        self.assertAlmostEqual(
            theoretical["entry_fill_price"],
            output["sessions"].loc[1, "low"],
        )
        self.assertAlmostEqual(
            theoretical["exit_fill_price"],
            output["sessions"].loc[3, "high"],
        )
        self.assertGreater(theoretical["profit"], 0)

    def test_theoretical_trade_does_not_buy_and_sell_in_same_session(self):
        output = run_paper_trading(
            np.array([100.0, 80.0]),
            self.strategy,
            PaperTradingConfig(
                spread_bps=0,
                slippage_bps=0,
                transaction_cost_bps=0,
                intraday_range_bps=0,
                max_exposure=1.0,
            ),
        )

        theoretical = output["theoretical"]
        self.assertTrue(theoretical["trade_available"])
        self.assertGreater(
            theoretical["exit_session"],
            theoretical["entry_session"],
        )


if __name__ == "__main__":
    unittest.main()
