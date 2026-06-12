"""Aplicacion Streamlit para el modelo tactico Monte Carlo de IVV."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from ivv_calibration_backtest import (
    BacktestConfig,
    calibrate_history,
    download_market_history,
    run_walk_forward_backtest,
)
from ivv_montecarlo_engine import (
    AggressiveTradingStrategy,
    MarketAssumptions,
    SimulationConfig,
    TradingStrategy,
    run_aggressive_trading,
    run_simulation,
    summarize_aggressive_trading,
)


st.set_page_config(
    page_title="IVV Tactical Monte Carlo",
    page_icon=None,
    layout="wide",
)


@st.cache_data(ttl=900)
def latest_ivv_price() -> float:
    data = yf.download(
        "IVV",
        period="5d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance no devolvio precios de IVV.")
    close = data["Close"]
    if getattr(close, "ndim", 1) > 1:
        close = close.iloc[:, 0]
    return float(close.dropna().iloc[-1])


@st.cache_data(ttl=21_600)
def market_history() -> object:
    return download_market_history()


def percentage(value: float) -> str:
    return f"{value:.2%}"


def show_plotly(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )


def display_calibration(calibration) -> None:
    st.subheader("Calibracion utilizada")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Retorno historico anual",
        percentage(calibration.diagnostics["annual_return"]),
    )
    col2.metric(
        "Volatilidad anual",
        percentage(calibration.diagnostics["annual_volatility"]),
    )
    col3.metric("Student-t df", f"{calibration.student_df:.2f}")
    col4.metric(
        "Correlacion IVV / USD-MXN",
        f"{calibration.diagnostics['ivv_fx_correlation']:.2f}",
    )
    st.caption(
        f"{calibration.observations} sesiones: "
        f"{calibration.start_date:%Y-%m-%d} a "
        f"{calibration.end_date:%Y-%m-%d}. "
        f"R2 macro diario: {calibration.diagnostics['macro_r_squared']:.2%}."
    )


def display_backtest(results, summary, confidence) -> None:
    st.subheader("Backtesting walk-forward")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        f"Cobertura precio ({confidence:.0%})",
        percentage(summary["price_coverage"]),
    )
    col2.metric("Error mediano MAPE", percentage(summary["price_mape"]))
    col3.metric(
        "Estrategia supera buy-and-hold",
        percentage(summary["strategy_outperformance"]),
    )
    col4.metric(
        "Posicion abierta al trimestre",
        percentage(summary["open_position_rate"]),
    )

    dates = results["end"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=results["price_p95"],
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=results["price_p05"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(138,165,199,0.35)",
            name="Intervalo P5-P95",
            hovertemplate="P5: USD %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=results["price_p50"],
            mode="lines+markers",
            line=dict(color="#174A8B", width=2),
            name="Mediana",
            hovertemplate="Mediana: USD %{y:,.2f}<extra></extra>",
        )
    )
    covered = results["price_covered"].map(
        {True: "Dentro del intervalo", False: "Fuera del intervalo"}
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=results["actual_final_price"],
            mode="markers",
            marker=dict(
                size=10,
                color=np.where(
                    results["price_covered"], "#2E7D32", "#C62828"
                ),
            ),
            text=covered,
            name="Precio observado",
            hovertemplate=(
                "Observado: USD %{y:,.2f}<br>%{text}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Intervalo pronosticado frente al precio observado",
        yaxis_title="IVV (USD)",
        xaxis_title="Fin de ventana",
    )
    show_plotly(fig)

    table = results[
        [
            "origin",
            "end",
            "actual_final_price",
            "price_p05",
            "price_p50",
            "price_p95",
            "price_covered",
            "actual_strategy_return",
            "actual_buy_hold_return",
        ]
    ].copy()
    st.dataframe(table, use_container_width=True, hide_index=True)


def display_aggressive_trading(prices, strategy) -> None:
    results = run_aggressive_trading(prices, strategy)
    summary = summarize_aggressive_trading(results)
    st.header(f"Trading agresivo con USD {strategy.capital_usd:,.0f}")
    st.caption(
        "Opera el capital completo y permite varios ciclos. El maximo teorico "
        "se calcula retrospectivamente; la salida real usa reglas ejecutables."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Capital final mediano",
        f"USD {summary['median_final_capital']:,.2f}",
        f"USD {summary['median_profit']:,.2f}",
    )
    col2.metric(
        "Probabilidad de utilidad",
        percentage(summary["probability_profit"]),
    )
    col3.metric(
        "Probabilidad de operar",
        percentage(summary["probability_trade"]),
    )
    col4.metric(
        "Posicion abierta al final",
        percentage(summary["probability_open"]),
    )

    timing1, timing2, timing3, timing4 = st.columns(4)
    entry_day = summary["median_entry_day"]
    best_day = summary["median_best_day"]
    timing1.metric(
        "Dia mediano de compra",
        "Sin entrada" if np.isnan(entry_day) else f"Sesion {entry_day:.0f}",
    )
    timing2.metric(
        "Dia mediano del mejor precio",
        "Sin entrada" if np.isnan(best_day) else f"Sesion {best_day:.0f}",
    )
    timing3.metric(
        "Beneficio maximo teorico mediano",
        f"USD {summary['median_theoretical_max_profit']:,.2f}",
    )
    timing4.metric(
        "Captura mediana del maximo",
        percentage(summary["median_capture_ratio"]),
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Capital final simulado")
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=results["final_capital"],
                nbinsx=70,
                marker_color="#8E24AA",
                opacity=0.78,
                name="Capital final",
                hovertemplate="Capital: USD %{x:,.2f}<br>Frecuencia: %{y}<extra></extra>",
            )
        )
        fig.add_vline(
            x=strategy.capital_usd,
            line_color="#222222",
            annotation_text="Capital inicial",
        )
        fig.add_vline(
            x=summary["median_final_capital"],
            line_color="#174A8B",
            line_width=2,
            annotation_text="Mediana",
        )
        fig.update_layout(
            xaxis_title="Capital final (USD)",
            yaxis_title="Trayectorias",
        )
        show_plotly(fig)

    with right:
        st.subheader("Momento de entrada y mejor precio")
        traded = results.loc[results["traded"]]
        fig = go.Figure()
        if traded.empty:
            fig.add_annotation(
                text="No se activaron compras",
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
            )
        else:
            fig.add_trace(
                go.Histogram(
                    x=traded["first_entry_day"],
                    nbinsx=20,
                    opacity=0.65,
                    name="Compra",
                    marker_color="#F26A21",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=traded["best_day_after_entry"],
                    nbinsx=20,
                    opacity=0.45,
                    name="Maximo posterior",
                    marker_color="#2E7D32",
                )
            )
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Sesion dentro de los proximos 3 meses",
            yaxis_title="Trayectorias",
        )
        show_plotly(fig)

    st.write(
        f"Intervalo del 90% para el capital final: "
        f"**USD {summary['p05_final_capital']:,.2f} a "
        f"USD {summary['p95_final_capital']:,.2f}**. "
        f"Promedio de operaciones iniciadas: **{summary['mean_trades']:.2f}**."
    )

    exit_counts = results[
        ["take_profit_exits", "stop_loss_exits", "trailing_exits"]
    ].sum()
    exit_table = exit_counts.rename(
        {
            "take_profit_exits": "Take-profit",
            "stop_loss_exits": "Stop-loss",
            "trailing_exits": "Trailing stop",
        }
    ).rename("salidas")
    st.dataframe(exit_table.to_frame(), use_container_width=True)
    st.warning(
        "El dia del maximo posterior no es una senal disponible en tiempo real. "
        "Sirve para medir cuanto potencial dejo sin capturar la regla de salida."
    )


def display_user_guide() -> None:
    with st.expander("Manual de uso y glosario", expanded=False):
        manual_tab, glossary_tab = st.tabs(("Manual basico", "Glosario"))

        with manual_tab:
            st.markdown(
                """
                1. **Configure el mercado.** Revise el precio inicial de IVV y
                   ajuste los supuestos economicos. Si activa la calibracion
                   historica, el modelo sustituye varios supuestos manuales con
                   estimaciones obtenidas de datos recientes.
                2. **Defina la estrategia.** Escriba las caidas que activan cada
                   compra, separadas por comas. Por ejemplo, `3, 6, 9, 12`
                   divide el capital en cuatro compras iguales.
                3. **Ajuste la simulacion.** Use 20,000 trayectorias para una
                   consulta normal. La misma semilla produce resultados
                   reproducibles cuando los demas parametros no cambian.
                4. **Ejecute la simulacion.** El boton principal genera escenarios
                   de 63 sesiones y compara la estrategia escalonada contra
                   comprar y mantener IVV desde hoy.
                5. **Interprete los resultados.** Observe la mediana, la
                   probabilidad de utilidad, el intervalo P5-P95 y la frecuencia
                   con la que se activan compras o quedan posiciones abiertas.
                6. **Pruebe el backtesting.** Este boton evalua ventanas historicas
                   sin usar informacion futura. Sirve para medir cobertura y
                   errores del modelo, no para garantizar resultados posteriores.
                7. **Revise el trading agresivo.** Configure capital, entrada y
                   salidas. Esta seccion usa el capital completo en cada ciclo y
                   por ello puede mostrar mayor variacion y riesgo.
                """
            )
            st.info(
                "Sugerencia: cambie un parametro a la vez y compare los "
                "resultados con la misma semilla."
            )

        with glossary_tab:
            st.markdown(
                """
                - **IVV:** ETF que busca seguir el indice S&P 500.
                - **Trayectoria:** posible evolucion simulada del precio durante
                  las 63 sesiones del horizonte.
                - **Monte Carlo:** metodo que repite miles de escenarios
                  aleatorios para estimar un rango de resultados.
                - **Retorno mediano:** resultado central; la mitad de las
                  trayectorias queda por encima y la otra mitad por debajo.
                - **P5-P95:** intervalo que contiene el 90% central de los
                  resultados simulados. No es una garantia.
                - **Volatilidad:** medida de la variacion esperada del precio. Un
                  valor mayor suele producir rangos de resultados mas amplios.
                - **Student-t:** distribucion usada para representar movimientos
                  extremos con mayor frecuencia que una distribucion normal.
                - **Drawdown o caida:** descenso desde un maximo previo que puede
                  activar una compra.
                - **Costo promedio:** precio medio pagado por las compras
                  ejecutadas.
                - **Take-profit:** venta al alcanzar una utilidad definida.
                - **Stop-loss:** venta al alcanzar una perdida maxima definida.
                - **Trailing stop:** salida que sigue al precio mientras sube y
                  vende cuando retrocede el porcentaje configurado.
                - **Buy and hold:** comprar IVV al inicio y mantenerlo hasta el
                  final del horizonte.
                - **Backtesting walk-forward:** evaluacion historica que calibra
                  con datos anteriores y prueba en el periodo siguiente.
                - **Semilla:** numero que permite repetir la misma secuencia
                  aleatoria y comparar configuraciones.
                - **Puntos base (pb):** unidad de costos; 100 pb equivalen a 1%.
                - **Posicion abierta:** compra que no alcanzo una regla de venta
                  antes de terminar el horizonte.
                """
            )


def main() -> None:
    st.title("IVV Tactical Monte Carlo")
    st.caption(
        "Compra escalonada en caidas, salida en recuperacion y escenarios "
        "geopoliticos con colas pesadas."
    )

    with st.sidebar:
        st.header("Mercado")
        try:
            default_price = latest_ivv_price()
        except Exception:
            default_price = 600.0
            st.warning("Se usa un precio provisional; revise la conexion.")

        initial_price = st.number_input(
            "Precio inicial IVV (USD)", min_value=1.0, value=default_price, step=1.0
        )
        expected_return = st.slider(
            "Rendimiento anual base", -20.0, 30.0, 8.0, 0.5
        )
        volatility = st.slider("Volatilidad anual", 5.0, 60.0, 18.0, 0.5)
        inflation = st.slider("Inflacion anual EE.UU.", 0.0, 15.0, 3.0, 0.1)
        interest_rate = st.slider("Tasa de interes de referencia", 0.0, 15.0, 4.0, 0.1)
        oil_change = st.slider("Cambio esperado petroleo a 3 meses", -40.0, 80.0, 0.0, 1.0)
        fx_change = st.slider("Cambio esperado USD/MXN a 3 meses", -20.0, 30.0, 0.0, 0.5)

        st.header("Estrategia")
        levels_text = st.text_input("Caidas para comprar (%)", "3, 6, 9, 12")
        take_profit = st.slider("Venta sobre costo promedio", 1.0, 20.0, 6.0, 0.5)
        transaction_cost = st.number_input(
            "Costo por operacion (pb)", min_value=0.0, value=10.0, step=1.0
        )

        st.header("Simulacion")
        paths = st.select_slider(
            "Trayectorias",
            options=[5_000, 10_000, 20_000, 50_000],
            value=20_000,
            help=(
                "20,000 es adecuado para uso normal. 50,000 requiere mas "
                "memoria y puede ser lento en servidores gratuitos."
            ),
        )
        student_df = st.slider(
            "Grados de libertad Student-t", 3.0, 20.0, 5.0, 0.5
        )
        seed = st.number_input("Semilla", min_value=0, value=42, step=1)
        st.header("Calibracion")
        use_calibration = st.checkbox(
            "Usar calibracion historica",
            value=True,
            help="Reemplaza retorno, volatilidad, colas y escenarios manuales.",
        )
        training_years = st.slider(
            "Ventana historica (anos)", 2, 10, 5, 1
        )
        backtest_windows = st.slider(
            "Ventanas walk-forward", 4, 24, 12, 1
        )
        st.header("Trading agresivo")
        aggressive_capital = st.number_input(
            "Capital agresivo (USD)",
            min_value=100.0,
            value=1_000.0,
            step=100.0,
        )
        aggressive_entry_label = st.radio(
            "Entrada agresiva",
            ("Esperar caida", "Comprar hoy"),
            horizontal=True,
        )
        aggressive_drawdown = st.slider(
            "Caida para entrar", 1.0, 20.0, 4.0, 0.5
        )
        aggressive_take_profit = st.slider(
            "Take-profit agresivo", 1.0, 25.0, 6.0, 0.5
        )
        aggressive_stop_loss = st.slider(
            "Stop-loss agresivo", 1.0, 20.0, 5.0, 0.5
        )
        aggressive_trailing = st.slider(
            "Trailing stop", 0.5, 15.0, 2.5, 0.5
        )
        aggressive_max_trades = st.slider(
            "Maximo de operaciones", 1, 8, 3, 1
        )
        run = st.button("Ejecutar simulacion", type="primary", use_container_width=True)
        run_backtest = st.button(
            "Ejecutar backtesting",
            use_container_width=True,
        )

    st.info(
        "Los pesos de escenario son 15% adverso, 70% central y 15% favorable. "
        "El intervalo del 90% se reporta entre P5 y P95."
    )
    display_user_guide()

    try:
        levels = tuple(
            sorted(float(item.strip()) / 100 for item in levels_text.split(","))
        )
        allocations = tuple(1 / len(levels) for _ in levels)
        strategy = TradingStrategy(
            drawdown_levels=levels,
            allocations=allocations,
            take_profit=take_profit / 100,
            transaction_cost_bps=transaction_cost,
        )
        strategy.validate()
    except Exception as error:
        st.error(f"Configuracion de estrategia invalida: {error}")
        return

    if run_backtest:
        try:
            with st.spinner("Descargando historia y ejecutando walk-forward..."):
                history = market_history()
                backtest_results, backtest_summary = run_walk_forward_backtest(
                    history,
                    strategy,
                    BacktestConfig(
                        training_days=training_years * 252,
                        horizon_days=63,
                        step_days=63,
                        paths=min(int(paths), 10_000),
                        confidence=0.90,
                        max_windows=backtest_windows,
                        seed=int(seed),
                    ),
                    inflation_annual=inflation / 100,
                )
            display_backtest(backtest_results, backtest_summary, 0.90)
        except Exception as error:
            st.error(f"No fue posible ejecutar el backtesting: {error}")

    if not run:
        st.subheader("Que evalua esta demo")
        st.write(
            "El motor genera 63 sesiones con volatilidad tipo GARCH, innovaciones "
            "Student-t y saltos distintos por escenario. En cada trayectoria aplica "
            "las compras configuradas y compara su resultado con comprar IVV hoy."
        )
        return

    try:
        assumptions = MarketAssumptions(
            initial_price=initial_price,
            expected_return_annual=expected_return / 100,
            volatility_annual=volatility / 100,
            inflation_annual=inflation / 100,
            interest_rate_annual=interest_rate / 100,
            oil_change_3m=oil_change / 100,
            fx_change_3m=fx_change / 100,
        )
        scenarios = None
        calibration = None
        calibrated_df = student_df
        if use_calibration:
            history = market_history()
            training = history.tail(training_years * 252)
            calibration = calibrate_history(training, inflation / 100)
            assumptions = replace(
                calibration.assumptions,
                initial_price=initial_price,
                inflation_annual=inflation / 100,
                oil_change_3m=oil_change / 100,
                fx_change_3m=fx_change / 100,
            )
            scenarios = calibration.scenarios
            calibrated_df = calibration.student_df

        config = SimulationConfig(
            paths=int(paths),
            days=63,
            confidence=0.90,
            student_df=calibrated_df,
            seed=int(seed),
        )
        with st.spinner("Simulando mercado y estrategia..."):
            if scenarios is None:
                output = run_simulation(assumptions, config, strategy)
            else:
                output = run_simulation(
                    assumptions,
                    config,
                    strategy,
                    scenarios,
                )
    except Exception as error:
        st.error(f"No fue posible ejecutar la simulacion: {error}")
        return

    results = output["results"]
    if calibration is not None:
        display_calibration(calibration)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Retorno mediano estrategia USD",
        percentage(results["strategy_return_usd"].median()),
    )
    col2.metric(
        "Probabilidad de utilidad",
        percentage((results["strategy_return_usd"] > 0).mean()),
    )
    col3.metric(
        "Probabilidad de activar compra",
        percentage((results["levels_triggered"] > 0).mean()),
    )
    col4.metric(
        "Posicion abierta al final",
        percentage(results["position_open_at_horizon"].mean()),
    )

    st.caption(
        "Rendimiento anual ajustado por los supuestos macro: "
        f"{percentage(output['adjusted_expected_return'])}"
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Distribucion de resultados")
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=results["strategy_return_usd"] * 100,
                nbinsx=70,
                opacity=0.75,
                name="Estrategia",
                marker_color="#174A8B",
                hovertemplate="Retorno: %{x:.2f}%<br>Frecuencia: %{y}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Histogram(
                x=results["buy_hold_return_usd"] * 100,
                nbinsx=70,
                opacity=0.35,
                name="Buy and hold",
                marker_color="#F26A21",
                hovertemplate="Retorno: %{x:.2f}%<br>Frecuencia: %{y}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color="#222222")
        fig.update_layout(
            barmode="overlay",
            xaxis_title="Rendimiento a 3 meses (%)",
            yaxis_title="Trayectorias",
        )
        show_plotly(fig)

    with right:
        st.subheader("Abanico de precios IVV")
        prices = output["prices"]
        sample_size = min(40, len(prices))
        rng = np.random.default_rng(config.seed)
        sample = prices[rng.choice(len(prices), sample_size, replace=False)]
        sessions = np.arange(prices.shape[1])
        p05 = np.quantile(prices, 0.05, axis=0)
        median = np.median(prices, axis=0)
        p95 = np.quantile(prices, 0.95, axis=0)
        fig = go.Figure()
        for path in sample:
            fig.add_trace(
                go.Scatter(
                    x=sessions,
                    y=path,
                    mode="lines",
                    line=dict(color="rgba(138,165,199,0.16)", width=1),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatter(
                x=sessions,
                y=p95,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sessions,
                y=p05,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(138,165,199,0.22)",
                name="Intervalo P5-P95",
                hovertemplate="P5: USD %{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sessions,
                y=median,
                mode="lines",
                line=dict(color="#174A8B", width=3),
                name="Mediana",
                hovertemplate="Mediana: USD %{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sessions,
                y=p95,
                mode="lines",
                line=dict(color="#2E7D32", width=1, dash="dash"),
                name="P95",
                hovertemplate="P95: USD %{y:,.2f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sessions,
                y=p05,
                mode="lines",
                line=dict(color="#C62828", width=1, dash="dash"),
                name="P5",
                hovertemplate="P5: USD %{y:,.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title="Sesion",
            yaxis_title="IVV (USD)",
        )
        show_plotly(fig)

    st.subheader("Percentiles")
    percentiles = output["percentiles"].copy()
    for column in percentiles.columns:
        if column != "final_price":
            percentiles[column] = percentiles[column].map(percentage)
        else:
            percentiles[column] = percentiles[column].map(lambda value: f"${value:,.2f}")
    st.dataframe(percentiles, use_container_width=True)

    st.subheader("Resultados por escenario")
    scenarios = output["scenarios"].copy()
    for column in (
        "weight",
        "median_strategy_usd",
        "probability_profit",
        "probability_buy",
        "probability_sale",
        "probability_open",
    ):
        scenarios[column] = scenarios[column].map(percentage)
    scenarios["median_final_price"] = scenarios["median_final_price"].map(
        lambda value: f"${value:,.2f}"
    )
    st.dataframe(scenarios, use_container_width=True, hide_index=True)

    aggressive_strategy = AggressiveTradingStrategy(
        capital_usd=aggressive_capital,
        entry_mode=(
            "immediate"
            if aggressive_entry_label == "Comprar hoy"
            else "buy_dip"
        ),
        entry_drawdown=aggressive_drawdown / 100,
        take_profit=aggressive_take_profit / 100,
        stop_loss=aggressive_stop_loss / 100,
        trailing_stop=aggressive_trailing / 100,
        cooldown_days=2,
        max_trades=aggressive_max_trades,
        transaction_cost_bps=transaction_cost,
    )
    st.divider()
    display_aggressive_trading(output["prices"], aggressive_strategy)

    st.warning(
        "Demo de investigacion, no recomendacion de inversion. Las sensibilidades "
        "macro y los shocks geopoliticos aun deben calibrarse y validarse mediante "
        "backtesting walk-forward antes de usar el resultado para operar."
    )


if __name__ == "__main__":
    main()
