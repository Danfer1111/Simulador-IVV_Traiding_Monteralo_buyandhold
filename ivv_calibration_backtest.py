"""Calibracion historica y backtesting walk-forward para el modelo IVV."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from ivv_montecarlo_engine import (
    MarketAssumptions,
    Scenario,
    SimulationConfig,
    TradingStrategy,
    run_simulation,
    run_trading_strategy,
)


TRADING_DAYS = 252
MARKET_TICKERS = {
    "ivv": "IVV",
    "oil": "CL=F",
    "rate_10y": "^TNX",
    "usdmxn": "MXN=X",
}


@dataclass(frozen=True)
class CalibrationResult:
    assumptions: MarketAssumptions
    scenarios: tuple[Scenario, ...]
    student_df: float
    observations: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class BacktestConfig:
    training_days: int = 756
    horizon_days: int = 63
    step_days: int = 63
    paths: int = 3_000
    confidence: float = 0.90
    max_windows: int = 20
    seed: int = 42


def _close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close", ticker] if ("Close", ticker) in data.columns else None
    else:
        close = data["Close"] if "Close" in data.columns else None
    if close is None:
        raise ValueError(f"No existe cierre para {ticker}.")
    return close.dropna().rename(ticker)


def download_market_history(
    start: str = "2007-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Descarga y alinea cierres diarios ajustados de las variables de mercado."""
    tickers = list(MARKET_TICKERS.values())
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
        group_by="column",
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance no devolvio datos para calibracion.")

    series = {
        name: _close_series(data, ticker)
        for name, ticker in MARKET_TICKERS.items()
    }
    history = pd.concat(series.values(), axis=1)
    history.columns = list(series)
    history = history.sort_index().ffill(limit=5).dropna()
    if len(history) < 1_000:
        raise RuntimeError("La historia alineada es insuficiente para calibrar.")
    return history


def _ridge_coefficients(
    target: pd.Series,
    features: pd.DataFrame,
    penalty: float = 8.0,
) -> tuple[np.ndarray, float]:
    aligned = pd.concat([target.rename("target"), features], axis=1).dropna()
    y = aligned.pop("target").to_numpy()
    x = aligned.to_numpy()
    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=1)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (x - means) / scales
    design = np.column_stack([np.ones(len(standardized)), standardized])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0
    coefficients = np.linalg.solve(
        design.T @ design + regularizer,
        design.T @ y,
    )
    prediction = design @ coefficients
    residual = y - prediction
    total_variance = np.sum((y - y.mean()) ** 2)
    r_squared = (
        1 - np.sum(residual**2) / total_variance if total_variance > 0 else 0.0
    )
    slopes = coefficients[1:] / scales
    return slopes, float(r_squared)


def _student_df_from_kurtosis(returns: pd.Series) -> float:
    excess = float(returns.kurt())
    if not np.isfinite(excess) or excess <= 0.05:
        return 20.0
    return float(np.clip(6 / excess + 4, 3.2, 20.0))


def _calibrate_scenarios(
    returns: pd.Series,
    base_return_annual: float,
) -> tuple[Scenario, ...]:
    trailing = (1 + returns).rolling(21).apply(np.prod, raw=True) - 1
    lower, upper = trailing.quantile([0.15, 0.85])
    labels = pd.Series("Central", index=returns.index)
    labels.loc[trailing <= lower] = "Adverso"
    labels.loc[trailing >= upper] = "Favorable"
    base_volatility = max(float(returns.std(ddof=1)), 1e-6)

    definitions = (
        ("Adverso", 0.15),
        ("Central", 0.70),
        ("Favorable", 0.15),
    )
    scenarios = []
    for name, probability in definitions:
        sample = returns.loc[labels == name].dropna()
        annual_return = float(sample.mean() * TRADING_DAYS)
        volatility_multiplier = float(
            np.clip(sample.std(ddof=1) / base_volatility, 0.55, 2.25)
        )
        threshold = max(2.5 * base_volatility, 0.015)
        if name == "Favorable":
            jumps = sample[sample >= threshold]
        elif name == "Adverso":
            jumps = sample[sample <= -threshold]
        else:
            jumps = sample[sample.abs() >= threshold]

        jump_probability = float(np.clip(len(jumps) / max(len(sample), 1), 0.002, 0.08))
        fallback_mean = {"Adverso": -0.02, "Central": -0.003, "Favorable": 0.015}[name]
        jump_mean = float(jumps.mean()) if len(jumps) >= 3 else fallback_mean
        jump_std = float(jumps.std(ddof=1)) if len(jumps) >= 3 else 0.01
        scenarios.append(
            Scenario(
                name=name,
                probability=probability,
                drift_adjustment_annual=float(
                    np.clip(annual_return - base_return_annual, -0.40, 0.40)
                ),
                volatility_multiplier=volatility_multiplier,
                jump_probability_daily=jump_probability,
                jump_mean=jump_mean,
                jump_std=max(jump_std, 0.005),
            )
        )
    return tuple(scenarios)


def calibrate_history(
    history: pd.DataFrame,
    inflation_annual: float = 0.03,
) -> CalibrationResult:
    """Calibra el modelo usando exclusivamente el tramo historico recibido."""
    required = {"ivv", "oil", "rate_10y", "usdmxn"}
    if not required.issubset(history.columns):
        raise ValueError(f"Faltan columnas: {sorted(required - set(history.columns))}")
    clean = history[list(required)].sort_index().ffill(limit=5).dropna()
    if len(clean) < 504:
        raise ValueError("Se requieren al menos dos anos de datos diarios.")

    ivv_returns = np.log(clean["ivv"]).diff()
    oil_returns = np.log(clean["oil"]).diff()
    fx_returns = np.log(clean["usdmxn"]).diff()
    rate_changes = clean["rate_10y"].diff() / 100
    features = pd.DataFrame(
        {
            "oil": oil_returns,
            "rate": rate_changes,
            "fx": fx_returns,
        }
    )
    slopes, r_squared = _ridge_coefficients(ivv_returns, features)
    annual_return = float(ivv_returns.mean() * TRADING_DAYS)
    annual_volatility = float(ivv_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    fx_volatility = float(fx_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    correlation = float(ivv_returns.corr(fx_returns))
    student_df = _student_df_from_kurtosis(ivv_returns.dropna())

    assumptions = MarketAssumptions(
        initial_price=float(clean["ivv"].iloc[-1]),
        expected_return_annual=float(np.clip(annual_return, -0.15, 0.25)),
        volatility_annual=float(np.clip(annual_volatility, 0.08, 0.50)),
        inflation_annual=inflation_annual,
        interest_rate_annual=float(clean["rate_10y"].iloc[-1] / 100),
        oil_change_3m=0.0,
        fx_change_3m=0.0,
        fx_volatility_annual=float(np.clip(fx_volatility, 0.05, 0.35)),
        inflation_reference=inflation_annual,
        rate_reference=float(clean["rate_10y"].iloc[-1] / 100),
        inflation_beta=0.0,
        rate_beta=float(np.clip(slopes[1], -3.0, 3.0)),
        oil_beta=float(np.clip(slopes[0], -1.0, 1.0)),
        ivv_fx_correlation=float(np.clip(correlation, -0.90, 0.90)),
    )
    scenarios = _calibrate_scenarios(ivv_returns.dropna(), annual_return)
    diagnostics = {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "student_df": student_df,
        "oil_beta_daily": float(slopes[0]),
        "rate_beta_daily": float(slopes[1]),
        "fx_beta_daily": float(slopes[2]),
        "ivv_fx_correlation": correlation,
        "macro_r_squared": r_squared,
        "excess_kurtosis": float(ivv_returns.kurt()),
    }
    return CalibrationResult(
        assumptions=assumptions,
        scenarios=scenarios,
        student_df=student_df,
        observations=len(clean),
        start_date=clean.index[0],
        end_date=clean.index[-1],
        diagnostics=diagnostics,
    )


def run_walk_forward_backtest(
    history: pd.DataFrame,
    strategy: TradingStrategy,
    config: BacktestConfig = BacktestConfig(),
    inflation_annual: float = 0.03,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evalua pronosticos y estrategia sin usar datos posteriores al origen."""
    minimum = config.training_days + config.horizon_days
    if len(history) < minimum:
        raise ValueError(f"Se requieren al menos {minimum} observaciones.")

    origins = list(
        range(
            config.training_days,
            len(history) - config.horizon_days,
            config.step_days,
        )
    )
    origins = origins[-config.max_windows :]
    rows = []

    for window_number, origin in enumerate(origins):
        training = history.iloc[origin - config.training_days : origin]
        future = history.iloc[origin : origin + config.horizon_days + 1]
        calibration = calibrate_history(training, inflation_annual)
        assumptions = calibration.assumptions
        simulation_config = SimulationConfig(
            paths=config.paths,
            days=config.horizon_days,
            confidence=config.confidence,
            student_df=calibration.student_df,
            seed=config.seed + window_number,
        )
        simulated = run_simulation(
            assumptions,
            simulation_config,
            strategy,
            calibration.scenarios,
        )
        simulated_results = simulated["results"]
        alpha = (1 - config.confidence) / 2
        price_low, price_median, price_high = np.quantile(
            simulated_results["final_price"],
            [alpha, 0.50, 1 - alpha],
        )
        strategy_low, strategy_median, strategy_high = np.quantile(
            simulated_results["strategy_return_usd"],
            [alpha, 0.50, 1 - alpha],
        )

        actual_prices = future["ivv"].to_numpy()[None, :]
        actual_fx = (
            future["usdmxn"].to_numpy() / future["usdmxn"].iloc[0]
        )[None, :]
        actual_strategy = run_trading_strategy(
            actual_prices,
            actual_fx,
            strategy,
        ).iloc[0]
        actual_final_price = float(future["ivv"].iloc[-1])
        rows.append(
            {
                "origin": future.index[0],
                "end": future.index[-1],
                "actual_final_price": actual_final_price,
                "price_p05": price_low,
                "price_p50": price_median,
                "price_p95": price_high,
                "price_covered": price_low <= actual_final_price <= price_high,
                "actual_strategy_return": actual_strategy["strategy_return_usd"],
                "strategy_p05": strategy_low,
                "strategy_p50": strategy_median,
                "strategy_p95": strategy_high,
                "strategy_covered": (
                    strategy_low
                    <= actual_strategy["strategy_return_usd"]
                    <= strategy_high
                ),
                "actual_buy_hold_return": actual_strategy["buy_hold_return_usd"],
                "actual_levels_triggered": actual_strategy["levels_triggered"],
                "actual_position_open": actual_strategy["position_open_at_horizon"],
                "calibrated_return": assumptions.expected_return_annual,
                "calibrated_volatility": assumptions.volatility_annual,
                "student_df": calibration.student_df,
            }
        )

    results = pd.DataFrame(rows)
    errors = results["price_p50"] - results["actual_final_price"]
    summary = {
        "windows": float(len(results)),
        "price_coverage": float(results["price_covered"].mean()),
        "strategy_coverage": float(results["strategy_covered"].mean()),
        "price_mape": float(
            (errors.abs() / results["actual_final_price"]).mean()
        ),
        "strategy_mean_return": float(results["actual_strategy_return"].mean()),
        "buy_hold_mean_return": float(results["actual_buy_hold_return"].mean()),
        "strategy_win_rate": float((results["actual_strategy_return"] > 0).mean()),
        "strategy_outperformance": float(
            (
                results["actual_strategy_return"]
                > results["actual_buy_hold_return"]
            ).mean()
        ),
        "open_position_rate": float(results["actual_position_open"].mean()),
    }
    return results, summary
