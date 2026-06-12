"""Motor Monte Carlo para una estrategia tactica de compra en caidas de IVV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class MarketAssumptions:
    initial_price: float
    expected_return_annual: float = 0.08
    volatility_annual: float = 0.18
    inflation_annual: float = 0.03
    interest_rate_annual: float = 0.04
    oil_change_3m: float = 0.0
    fx_change_3m: float = 0.0
    fx_volatility_annual: float = 0.12
    inflation_reference: float = 0.025
    rate_reference: float = 0.04
    inflation_beta: float = -0.60
    rate_beta: float = -0.75
    oil_beta: float = -0.12
    ivv_fx_correlation: float = -0.25


@dataclass(frozen=True)
class SimulationConfig:
    paths: int = 20_000
    days: int = 63
    confidence: float = 0.90
    student_df: float = 5.0
    seed: int = 42
    garch_alpha: float = 0.08
    garch_beta: float = 0.90


@dataclass(frozen=True)
class TradingStrategy:
    drawdown_levels: tuple[float, ...] = (0.03, 0.06, 0.09, 0.12)
    allocations: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
    take_profit: float = 0.06
    transaction_cost_bps: float = 10.0

    def validate(self) -> None:
        if len(self.drawdown_levels) != len(self.allocations):
            raise ValueError("Cada nivel de compra necesita una asignacion.")
        if any(level <= 0 for level in self.drawdown_levels):
            raise ValueError("Los niveles de drawdown deben ser positivos.")
        if tuple(sorted(self.drawdown_levels)) != self.drawdown_levels:
            raise ValueError("Los niveles de drawdown deben ser ascendentes.")
        if any(allocation <= 0 for allocation in self.allocations):
            raise ValueError("Las asignaciones deben ser positivas.")
        if sum(self.allocations) > 1.0 + 1e-9:
            raise ValueError("Las asignaciones no pueden superar 100% del capital.")
        if self.take_profit <= 0:
            raise ValueError("El objetivo de salida debe ser positivo.")


@dataclass(frozen=True)
class Scenario:
    name: str
    probability: float
    drift_adjustment_annual: float
    volatility_multiplier: float
    jump_probability_daily: float
    jump_mean: float
    jump_std: float


DEFAULT_SCENARIOS = (
    Scenario(
        name="Adverso",
        probability=0.15,
        drift_adjustment_annual=-0.18,
        volatility_multiplier=1.65,
        jump_probability_daily=0.035,
        jump_mean=-0.025,
        jump_std=0.018,
    ),
    Scenario(
        name="Central",
        probability=0.70,
        drift_adjustment_annual=0.00,
        volatility_multiplier=1.00,
        jump_probability_daily=0.006,
        jump_mean=-0.005,
        jump_std=0.010,
    ),
    Scenario(
        name="Favorable",
        probability=0.15,
        drift_adjustment_annual=0.12,
        volatility_multiplier=0.80,
        jump_probability_daily=0.010,
        jump_mean=0.018,
        jump_std=0.012,
    ),
)


def _validate_inputs(
    assumptions: MarketAssumptions,
    config: SimulationConfig,
    strategy: TradingStrategy,
    scenarios: tuple[Scenario, ...],
) -> None:
    strategy.validate()
    if assumptions.initial_price <= 0:
        raise ValueError("El precio inicial debe ser positivo.")
    if assumptions.volatility_annual <= 0:
        raise ValueError("La volatilidad debe ser positiva.")
    if config.paths < 1_000:
        raise ValueError("Use al menos 1,000 trayectorias.")
    if config.days <= 0:
        raise ValueError("El horizonte debe ser positivo.")
    if config.student_df <= 2:
        raise ValueError("Student-t requiere mas de 2 grados de libertad.")
    if not 0 < config.confidence < 1:
        raise ValueError("El nivel de confianza debe estar entre 0 y 1.")
    if not np.isclose(sum(item.probability for item in scenarios), 1.0):
        raise ValueError("Las probabilidades de escenarios deben sumar 100%.")
    if config.garch_alpha + config.garch_beta >= 1:
        raise ValueError("GARCH requiere alpha + beta menor que 1.")


def adjusted_expected_return(assumptions: MarketAssumptions) -> float:
    """Ajusta el retorno anual esperado con sensibilidades macro configurables."""
    inflation_effect = assumptions.inflation_beta * (
        assumptions.inflation_annual - assumptions.inflation_reference
    )
    rate_effect = assumptions.rate_beta * (
        assumptions.interest_rate_annual - assumptions.rate_reference
    )
    oil_effect = assumptions.oil_beta * assumptions.oil_change_3m
    return (
        assumptions.expected_return_annual
        + inflation_effect
        + rate_effect
        + oil_effect
    )


def simulate_market(
    assumptions: MarketAssumptions,
    config: SimulationConfig,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simula precios de IVV y USD/MXN con colas pesadas, GARCH y saltos."""
    _validate_inputs(assumptions, config, TradingStrategy(), scenarios)
    rng = np.random.default_rng(config.seed)
    probabilities = np.array([item.probability for item in scenarios])
    scenario_ids = rng.choice(len(scenarios), size=config.paths, p=probabilities)

    prices = np.empty((config.paths, config.days + 1), dtype=np.float64)
    fx_index = np.empty_like(prices)
    prices[:, 0] = assumptions.initial_price
    fx_index[:, 0] = 1.0

    base_return = adjusted_expected_return(assumptions)
    scenario_drift = np.array(
        [scenarios[index].drift_adjustment_annual for index in scenario_ids]
    )
    vol_multiplier = np.array(
        [scenarios[index].volatility_multiplier for index in scenario_ids]
    )
    jump_probability = np.array(
        [scenarios[index].jump_probability_daily for index in scenario_ids]
    )
    jump_mean = np.array([scenarios[index].jump_mean for index in scenario_ids])
    jump_std = np.array([scenarios[index].jump_std for index in scenario_ids])

    daily_variance = (
        assumptions.volatility_annual * vol_multiplier / np.sqrt(TRADING_DAYS)
    ) ** 2
    long_run_variance = daily_variance.copy()
    previous_shock = np.zeros(config.paths)
    t_scale = np.sqrt((config.student_df - 2) / config.student_df)
    fx_daily_drift = np.log1p(assumptions.fx_change_3m) / config.days
    fx_daily_vol = assumptions.fx_volatility_annual / np.sqrt(TRADING_DAYS)
    correlation = np.clip(assumptions.ivv_fx_correlation, -0.99, 0.99)

    for day in range(1, config.days + 1):
        daily_variance = (
            (1 - config.garch_alpha - config.garch_beta) * long_run_variance
            + config.garch_alpha * previous_shock**2
            + config.garch_beta * daily_variance
        )
        ivv_innovation = rng.standard_t(config.student_df, config.paths) * t_scale
        continuous_shock = np.sqrt(daily_variance) * ivv_innovation
        jump_occurs = rng.random(config.paths) < jump_probability
        jumps = jump_occurs * rng.normal(jump_mean, jump_std)
        daily_drift = (base_return + scenario_drift) / TRADING_DAYS
        log_return = daily_drift - 0.5 * daily_variance + continuous_shock + jumps
        prices[:, day] = prices[:, day - 1] * np.exp(log_return)
        previous_shock = continuous_shock + jumps

        independent_fx = rng.standard_t(config.student_df, config.paths) * t_scale
        fx_innovation = (
            correlation * ivv_innovation
            + np.sqrt(1 - correlation**2) * independent_fx
        )
        fx_log_return = (
            fx_daily_drift - 0.5 * fx_daily_vol**2 + fx_daily_vol * fx_innovation
        )
        fx_index[:, day] = fx_index[:, day - 1] * np.exp(fx_log_return)

    return prices, fx_index, scenario_ids


def run_trading_strategy(
    prices: np.ndarray,
    fx_index: np.ndarray,
    strategy: TradingStrategy,
) -> pd.DataFrame:
    """Ejecuta una campana de compras escalonadas y una salida por recuperacion."""
    strategy.validate()
    paths, observations = prices.shape
    cash = np.ones(paths)
    shares = np.zeros(paths)
    invested_cost = np.zeros(paths)
    peak = prices[:, 0].copy()
    completed = np.zeros(paths, dtype=bool)
    sold_before_horizon = np.zeros(paths, dtype=bool)
    sale_day = np.full(paths, observations - 1, dtype=int)
    levels_triggered = np.zeros(paths, dtype=int)
    max_capital_used = np.zeros(paths)
    transaction_rate = strategy.transaction_cost_bps / 10_000

    level_done = np.zeros((paths, len(strategy.drawdown_levels)), dtype=bool)

    for day in range(1, observations):
        current = prices[:, day]
        peak = np.maximum(peak, current)
        drawdown = 1 - current / peak

        for level_index, (level, allocation) in enumerate(
            zip(strategy.drawdown_levels, strategy.allocations)
        ):
            buy = (
                (drawdown >= level)
                & ~level_done[:, level_index]
                & ~completed
                & (cash >= allocation)
            )
            if not buy.any():
                continue
            gross = np.full(paths, allocation)
            fee = gross * transaction_rate
            shares_bought = np.divide(
                gross - fee,
                current,
                out=np.zeros(paths),
                where=buy,
            )
            shares += np.where(buy, shares_bought, 0)
            cash -= np.where(buy, gross, 0)
            invested_cost += np.where(buy, gross, 0)
            level_done[buy, level_index] = True
            levels_triggered[buy] += 1
            max_capital_used = np.maximum(max_capital_used, invested_cost)

        average_cost = np.divide(
            invested_cost,
            shares,
            out=np.zeros(paths),
            where=shares > 0,
        )
        sell = (
            (shares > 0)
            & ~completed
            & (current >= average_cost * (1 + strategy.take_profit))
        )
        if sell.any():
            proceeds = shares * current * (1 - transaction_rate)
            cash += np.where(sell, proceeds, 0)
            shares = np.where(sell, 0, shares)
            completed[sell] = True
            sold_before_horizon[sell] = True
            sale_day[sell] = day

    final_price = prices[:, -1]
    final_fx = fx_index[:, -1]
    terminal_proceeds = shares * final_price * (1 - transaction_rate)
    terminal_value_usd = cash + terminal_proceeds
    strategy_return_usd = terminal_value_usd - 1
    strategy_return_mxn = terminal_value_usd * final_fx - 1
    buy_hold_return_usd = final_price / prices[:, 0] - 1
    buy_hold_return_mxn = (1 + buy_hold_return_usd) * final_fx - 1

    return pd.DataFrame(
        {
            "strategy_return_usd": strategy_return_usd,
            "strategy_return_mxn": strategy_return_mxn,
            "buy_hold_return_usd": buy_hold_return_usd,
            "buy_hold_return_mxn": buy_hold_return_mxn,
            "final_price": final_price,
            "fx_change": final_fx - 1,
            "levels_triggered": levels_triggered,
            "capital_used": max_capital_used,
            "sold_before_horizon": sold_before_horizon,
            "sale_day": sale_day,
            "position_open_at_horizon": shares > 0,
        }
    )


def summarize_results(
    results: pd.DataFrame,
    scenario_ids: np.ndarray,
    config: SimulationConfig,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construye percentiles globales y resultados separados por escenario."""
    alpha = (1 - config.confidence) / 2
    percentiles = sorted({alpha, 0.15, 0.50, 0.85, 1 - alpha})
    labels = [f"P{round(value * 100):02d}" for value in percentiles]
    metrics = {}

    for column in (
        "strategy_return_usd",
        "strategy_return_mxn",
        "buy_hold_return_usd",
        "final_price",
    ):
        metrics[column] = results[column].quantile(percentiles).to_numpy()

    percentile_table = pd.DataFrame(metrics, index=labels)

    scenario_rows = []
    for index, scenario in enumerate(scenarios):
        subset = results.loc[scenario_ids == index]
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "weight": scenario.probability,
                "paths": len(subset),
                "median_strategy_usd": subset["strategy_return_usd"].median(),
                "probability_profit": (subset["strategy_return_usd"] > 0).mean(),
                "probability_buy": (subset["levels_triggered"] > 0).mean(),
                "probability_sale": subset["sold_before_horizon"].mean(),
                "probability_open": subset["position_open_at_horizon"].mean(),
                "median_final_price": subset["final_price"].median(),
            }
        )

    return percentile_table, pd.DataFrame(scenario_rows)


def run_simulation(
    assumptions: MarketAssumptions,
    config: SimulationConfig,
    strategy: TradingStrategy,
    scenarios: tuple[Scenario, ...] = DEFAULT_SCENARIOS,
) -> dict[str, object]:
    """Punto de entrada para aplicaciones, reportes y pruebas."""
    _validate_inputs(assumptions, config, strategy, scenarios)
    prices, fx_index, scenario_ids = simulate_market(
        assumptions, config, scenarios
    )
    results = run_trading_strategy(prices, fx_index, strategy)
    percentile_table, scenario_table = summarize_results(
        results, scenario_ids, config, scenarios
    )
    return {
        "prices": prices,
        "fx_index": fx_index,
        "scenario_ids": scenario_ids,
        "results": results,
        "percentiles": percentile_table,
        "scenarios": scenario_table,
        "adjusted_expected_return": adjusted_expected_return(assumptions),
    }
