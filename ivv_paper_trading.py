"""Simulador operativo de paper trading sin conexion a un broker."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PaperTradingConfig:
    initial_capital_usd: float = 10_000.0
    spread_bps: float = 4.0
    slippage_bps: float = 2.0
    transaction_cost_bps: float = 10.0
    limit_offset_bps: float = 5.0
    order_expiry_sessions: int = 2
    intraday_range_bps: float = 35.0
    max_exposure: float = 0.80
    max_portfolio_loss: float = 0.10
    allow_fractional_shares: bool = True

    def validate(self) -> None:
        if self.initial_capital_usd <= 0:
            raise ValueError("El capital inicial debe ser positivo.")
        if min(
            self.spread_bps,
            self.slippage_bps,
            self.transaction_cost_bps,
            self.limit_offset_bps,
            self.intraday_range_bps,
        ) < 0:
            raise ValueError("Los costos y diferenciales no pueden ser negativos.")
        if self.order_expiry_sessions < 1:
            raise ValueError("La vigencia debe ser de al menos una sesion.")
        if not 0 < self.max_exposure <= 1:
            raise ValueError("La exposicion maxima debe estar entre 0% y 100%.")
        if not 0 < self.max_portfolio_loss < 1:
            raise ValueError("La perdida maxima debe estar entre 0% y 100%.")


@dataclass(frozen=True)
class PaperTradingStrategy:
    drawdown_levels: tuple[float, ...] = (0.03, 0.06, 0.09, 0.12)
    allocations: tuple[float, ...] = (0.20, 0.20, 0.20, 0.20)
    take_profit: float = 0.06
    stop_loss: float = 0.05
    trailing_stop: float = 0.03

    def validate(self) -> None:
        if len(self.drawdown_levels) != len(self.allocations):
            raise ValueError("Cada nivel necesita una asignacion de capital.")
        if not self.drawdown_levels:
            raise ValueError("Se necesita al menos un nivel de compra.")
        if tuple(sorted(self.drawdown_levels)) != self.drawdown_levels:
            raise ValueError("Los niveles de compra deben ser ascendentes.")
        if any(not 0 < value < 1 for value in self.drawdown_levels):
            raise ValueError("Las caidas deben estar entre 0% y 100%.")
        if any(value <= 0 for value in self.allocations):
            raise ValueError("Las asignaciones deben ser positivas.")
        if sum(self.allocations) > 1 + 1e-9:
            raise ValueError("Las asignaciones no pueden superar 100%.")
        if any(
            not 0 < value < 1
            for value in (self.take_profit, self.stop_loss, self.trailing_stop)
        ):
            raise ValueError("Las reglas de salida deben estar entre 0% y 100%.")


def _session_bars(prices: np.ndarray, intraday_range_bps: float) -> pd.DataFrame:
    close = np.asarray(prices, dtype=float)
    if close.ndim != 1 or len(close) < 2:
        raise ValueError("Se requiere una trayectoria de al menos dos precios.")
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("Todos los precios deben ser positivos y finitos.")

    open_price = np.r_[close[0], close[:-1]]
    move = np.abs(close / open_price - 1)
    minimum_range = intraday_range_bps / 10_000
    excursion = np.maximum(minimum_range, move * 0.35)
    high = np.maximum(open_price, close) * (1 + excursion)
    low = np.minimum(open_price, close) * (1 - excursion)
    return pd.DataFrame(
        {
            "session": np.arange(len(close)),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def _theoretical_hindsight_trade(
    bars: pd.DataFrame,
    config: PaperTradingConfig,
) -> dict[str, float | int | bool]:
    """Calcula la mejor compra y venta posterior con conocimiento del futuro."""
    fee_rate = config.transaction_cost_bps / 10_000
    half_spread = config.spread_bps / 20_000
    slippage = config.slippage_bps / 10_000
    deployable_capital = config.initial_capital_usd * config.max_exposure

    best: dict[str, float | int | bool] = {
        "trade_available": False,
        "entry_session": -1,
        "exit_session": -1,
        "entry_market_low": np.nan,
        "exit_market_high": np.nan,
        "entry_fill_price": np.nan,
        "exit_fill_price": np.nan,
        "shares": 0.0,
        "profit": 0.0,
        "return": 0.0,
        "final_equity": config.initial_capital_usd,
    }
    best_profit = 0.0

    for entry_session in range(len(bars) - 1):
        market_low = float(bars.loc[entry_session, "low"])
        entry_fill = market_low * (1 + half_spread) * (1 + slippage)
        raw_shares = deployable_capital / (entry_fill * (1 + fee_rate))
        shares = (
            raw_shares
            if config.allow_fractional_shares
            else float(np.floor(raw_shares))
        )
        if shares <= 0:
            continue

        exit_session = int(
            bars.loc[entry_session + 1 :, "high"].idxmax()
        )
        market_high = float(bars.loc[exit_session, "high"])
        exit_fill = market_high * (1 - half_spread) * (1 - slippage)
        purchase_cost = shares * entry_fill * (1 + fee_rate)
        sale_proceeds = shares * exit_fill * (1 - fee_rate)
        profit = sale_proceeds - purchase_cost
        if profit <= best_profit:
            continue

        best_profit = profit
        best = {
            "trade_available": True,
            "entry_session": entry_session,
            "exit_session": exit_session,
            "entry_market_low": market_low,
            "exit_market_high": market_high,
            "entry_fill_price": entry_fill,
            "exit_fill_price": exit_fill,
            "shares": shares,
            "profit": profit,
            "return": profit / config.initial_capital_usd,
            "final_equity": config.initial_capital_usd + profit,
        }

    return best


def run_paper_trading(
    prices: np.ndarray,
    strategy: PaperTradingStrategy,
    config: PaperTradingConfig,
) -> dict[str, object]:
    """Ejecuta ordenes simuladas y devuelve sesiones, bitacora y resumen."""
    strategy.validate()
    config.validate()
    bars = _session_bars(prices, config.intraday_range_bps)
    fee_rate = config.transaction_cost_bps / 10_000
    half_spread = config.spread_bps / 20_000
    slippage = config.slippage_bps / 10_000
    limit_offset = config.limit_offset_bps / 10_000

    cash = config.initial_capital_usd
    shares = 0.0
    cost_basis = 0.0
    peak = float(bars.loc[0, "close"])
    high_since_entry = 0.0
    pending_order: dict[str, float | int] | None = None
    completed_levels: set[int] = set()
    rejected_levels: set[int] = set()
    halted = False
    order_sequence = 0
    events: list[dict[str, object]] = []
    session_rows: list[dict[str, object]] = []

    def record(
        session: int,
        event: str,
        side: str = "",
        status: str = "",
        reason: str = "",
        limit_price: float = np.nan,
        fill_price: float = np.nan,
        quantity: float = 0.0,
    ) -> None:
        bid = float(bars.loc[session, "close"]) * (1 - half_spread)
        equity = cash + shares * bid
        events.append(
            {
                "session": session,
                "event": event,
                "side": side,
                "status": status,
                "reason": reason,
                "limit_price": limit_price,
                "fill_price": fill_price,
                "quantity": quantity,
                "cash": cash,
                "position_shares": shares,
                "equity": equity,
            }
        )

    for session, bar in bars.iterrows():
        open_price = float(bar["open"])
        high = float(bar["high"])
        low = float(bar["low"])
        close = float(bar["close"])
        bid = close * (1 - half_spread)
        ask = close * (1 + half_spread)

        if pending_order is not None and session > pending_order["submitted_session"]:
            order_age = session - int(pending_order["submitted_session"])
            limit_price = float(pending_order["limit_price"])
            if low <= limit_price:
                opening_ask = open_price * (1 + half_spread)
                fill_price = min(limit_price, opening_ask * (1 + slippage))
                quantity = float(pending_order["quantity"])
                total_cost = quantity * fill_price * (1 + fee_rate)
                if total_cost <= cash + 1e-9:
                    cash -= total_cost
                    shares += quantity
                    cost_basis += total_cost
                    high_since_entry = max(high_since_entry, fill_price, high)
                    level_index = int(pending_order["level_index"])
                    completed_levels.add(level_index)
                    record(
                        session,
                        "Orden ejecutada",
                        "Compra",
                        "Ejecutada",
                        f"Nivel {strategy.drawdown_levels[level_index]:.1%}",
                        limit_price,
                        fill_price,
                        quantity,
                    )
                else:
                    record(
                        session,
                        "Orden rechazada",
                        "Compra",
                        "Rechazada",
                        "Efectivo insuficiente al ejecutar",
                        limit_price,
                    )
                pending_order = None
            elif order_age >= config.order_expiry_sessions:
                record(
                    session,
                    "Orden vencida",
                    "Compra",
                    "Vencida",
                    "No se alcanzo el precio limite",
                    limit_price,
                )
                pending_order = None

        if shares > 0:
            average_cost = cost_basis / shares
            previous_high = high_since_entry
            high_since_entry = max(high_since_entry, high)
            stop_price = average_cost * (1 - strategy.stop_loss)
            trailing_price = high_since_entry * (1 - strategy.trailing_stop)
            target_price = average_cost * (1 + strategy.take_profit)
            exit_reason = ""
            trigger_price = np.nan

            if low <= stop_price:
                exit_reason = "Stop-loss"
                trigger_price = stop_price
            elif previous_high > average_cost and low <= trailing_price:
                exit_reason = "Trailing stop"
                trigger_price = trailing_price
            elif high >= target_price:
                exit_reason = "Take-profit"
                trigger_price = target_price

            if exit_reason:
                opening_bid = open_price * (1 - half_spread)
                fill_price = min(trigger_price, opening_bid) * (1 - slippage)
                quantity = shares
                proceeds = quantity * fill_price * (1 - fee_rate)
                cash += proceeds
                shares = 0.0
                cost_basis = 0.0
                high_since_entry = 0.0
                record(
                    session,
                    "Orden ejecutada",
                    "Venta",
                    "Ejecutada",
                    exit_reason,
                    trigger_price,
                    fill_price,
                    quantity,
                )

        equity = cash + shares * bid
        portfolio_return = equity / config.initial_capital_usd - 1
        if portfolio_return <= -config.max_portfolio_loss and not halted:
            if pending_order is not None:
                record(
                    session,
                    "Orden cancelada",
                    "Compra",
                    "Cancelada",
                    "Limite maximo de perdida",
                    float(pending_order["limit_price"]),
                )
                pending_order = None
            if shares > 0:
                fill_price = bid * (1 - slippage)
                quantity = shares
                cash += quantity * fill_price * (1 - fee_rate)
                shares = 0.0
                cost_basis = 0.0
                high_since_entry = 0.0
                record(
                    session,
                    "Liquidacion de riesgo",
                    "Venta",
                    "Ejecutada",
                    "Perdida maxima de cartera",
                    fill_price=fill_price,
                    quantity=quantity,
                )
            halted = True
            record(
                session,
                "Sistema detenido",
                status="Bloqueado",
                reason="Se alcanzo la perdida maxima permitida",
            )

        peak = max(peak, close)
        drawdown = 1 - close / peak
        if not halted and pending_order is None:
            for level_index, (level, allocation) in enumerate(
                zip(strategy.drawdown_levels, strategy.allocations)
            ):
                if (
                    level_index in completed_levels
                    or level_index in rejected_levels
                    or drawdown < level
                ):
                    continue

                budget = config.initial_capital_usd * allocation
                limit_price = ask * (1 - limit_offset)
                raw_quantity = budget / (limit_price * (1 + fee_rate))
                quantity = (
                    raw_quantity
                    if config.allow_fractional_shares
                    else float(np.floor(raw_quantity))
                )
                projected_exposure = shares * close + quantity * limit_price
                exposure_limit = config.initial_capital_usd * config.max_exposure
                required_cash = quantity * limit_price * (1 + fee_rate)
                rejection_reason = ""
                if quantity <= 0:
                    rejection_reason = "Capital insuficiente para una participacion"
                elif projected_exposure > exposure_limit + 1e-9:
                    rejection_reason = "Supera la exposicion maxima"
                elif required_cash > cash + 1e-9:
                    rejection_reason = "Efectivo insuficiente"

                if rejection_reason:
                    rejected_levels.add(level_index)
                    record(
                        session,
                        "Orden rechazada",
                        "Compra",
                        "Rechazada",
                        rejection_reason,
                        limit_price,
                        quantity=quantity,
                    )
                else:
                    order_sequence += 1
                    pending_order = {
                        "order_id": order_sequence,
                        "submitted_session": session,
                        "level_index": level_index,
                        "limit_price": limit_price,
                        "quantity": quantity,
                    }
                    record(
                        session,
                        "Orden creada",
                        "Compra",
                        "Pendiente",
                        f"Caida de {drawdown:.2%}; nivel {level:.1%}",
                        limit_price,
                        quantity=quantity,
                    )
                break

        average_cost = cost_basis / shares if shares > 0 else np.nan
        equity = cash + shares * bid
        session_rows.append(
            {
                "session": session,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "bid": bid,
                "ask": ask,
                "drawdown": drawdown,
                "cash": cash,
                "position_shares": shares,
                "average_cost": average_cost,
                "equity": equity,
                "return": equity / config.initial_capital_usd - 1,
                "pending_order": pending_order is not None,
                "risk_halted": halted,
            }
        )

    if pending_order is not None:
        record(
            len(bars) - 1,
            "Orden cancelada",
            "Compra",
            "Cancelada",
            "Fin de la simulacion",
            float(pending_order["limit_price"]),
        )

    sessions = pd.DataFrame(session_rows)
    event_log = pd.DataFrame(events)
    final_equity = float(sessions.iloc[-1]["equity"])
    theoretical = _theoretical_hindsight_trade(bars, config)
    actual_profit = final_equity - config.initial_capital_usd
    theoretical_profit = float(theoretical["profit"])
    capture_ratio = (
        max(actual_profit, 0) / theoretical_profit
        if theoretical_profit > 0
        else 0.0
    )
    fills = (
        event_log.loc[event_log["status"] == "Ejecutada"]
        if not event_log.empty
        else event_log
    )
    summary = {
        "initial_capital": config.initial_capital_usd,
        "final_equity": final_equity,
        "return": final_equity / config.initial_capital_usd - 1,
        "max_drawdown": float(
            (sessions["equity"] / sessions["equity"].cummax() - 1).min()
        ),
        "orders_created": int(
            (event_log["event"] == "Orden creada").sum()
            if not event_log.empty
            else 0
        ),
        "orders_filled": int(len(fills)),
        "orders_expired": int(
            (event_log["status"] == "Vencida").sum()
            if not event_log.empty
            else 0
        ),
        "orders_rejected": int(
            (event_log["status"] == "Rechazada").sum()
            if not event_log.empty
            else 0
        ),
        "risk_halted": halted,
        "open_position": bool(sessions.iloc[-1]["position_shares"] > 0),
        "actual_profit": actual_profit,
        "theoretical_profit": theoretical_profit,
        "capture_ratio": capture_ratio,
    }
    return {
        "sessions": sessions,
        "events": event_log,
        "summary": summary,
        "theoretical": theoretical,
    }
