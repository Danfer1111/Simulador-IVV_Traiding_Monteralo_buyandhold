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
        "Retorno historico anual (%)",
        percentage(calibration.diagnostics["annual_return"]),
    )
    col2.metric(
        "Volatilidad anual (%)",
        percentage(calibration.diagnostics["annual_volatility"]),
    )
    col3.metric("Student-t df (sin unidad)", f"{calibration.student_df:.2f}")
    col4.metric(
        "Correlacion IVV / USD-MXN (-1 a 1)",
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
    col2.metric("Error mediano MAPE (%)", percentage(summary["price_mape"]))
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
    table = table.rename(
        columns={
            "origin": "Inicio de prueba",
            "end": "Fin de prueba",
            "actual_final_price": "Precio observado (USD)",
            "price_p05": "Precio P5 (USD)",
            "price_p50": "Precio mediano (USD)",
            "price_p95": "Precio P95 (USD)",
            "price_covered": "Dentro de P5-P95",
            "actual_strategy_return": "Retorno estrategia (%)",
            "actual_buy_hold_return": "Retorno comprar y mantener (%)",
        }
    )
    for column in ("Retorno estrategia (%)", "Retorno comprar y mantener (%)"):
        table[column] = table[column].map(percentage)
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
    ).rename("Cantidad de salidas")
    st.dataframe(exit_table.to_frame(), use_container_width=True)
    st.warning(
        "El dia del maximo posterior no es una senal disponible en tiempo real. "
        "Sirve para medir cuanto potencial dejo sin capturar la regla de salida."
    )


def display_user_guide() -> None:
    with st.expander("Manual de uso y glosario", expanded=False):
        start_tab, advanced_tab, results_tab, glossary_tab = st.tabs(
            (
                "Empieza aqui",
                "Modo avanzado",
                "Como leer los resultados",
                "Glosario",
            )
        )

        with start_tab:
            st.markdown(
                """
                Esta pagina **no compra ni vende nada**. Solo crea miles de futuros
                posibles para ayudarte a entender el riesgo de una estrategia.

                **La forma mas facil de usarla**

                1. En la barra izquierda elige **Basico**.
                2. Revisa el precio de IVV. La pagina intenta obtener el precio
                   mas reciente automaticamente.
                3. Elige un perfil:
                   **Conservador** espera caidas mayores,
                   **Moderado** usa valores intermedios y
                   **Agresivo** compra ante caidas menores y acepta mas riesgo.
                4. Presiona **Ejecutar simulacion** una sola vez.
                5. Espera a que desaparezca el mensaje de carga.
                6. Empieza leyendo las cuatro tarjetas grandes. Despues revisa
                   las graficas.

                **No sabes que elegir?** Empieza con **Moderado**. Cambia solamente
                un ajuste cada vez para entender que efecto tiene.

                **Modo Avanzado**

                Usalo cuando ya conozcas conceptos como volatilidad, percentiles,
                costos de operacion y stop-loss. Este modo permite modificar todos
                los supuestos, ejecutar pruebas historicas y configurar trading
                agresivo.
                """
            )

            st.warning(
                "Una simulacion no sabe que ocurrira manana. Sirve para explorar "
                "posibilidades, no para prometer ganancias."
            )

        with advanced_tab:
            st.markdown(
                """
                **Que significa cada unidad**

                - **% (porcentaje):** una parte de cada 100. Por ejemplo, 6%
                  significa 6 por cada 100. Un retorno de -4% indica perdida.
                - **USD:** dolares estadounidenses. Se usa para precios, capital,
                  ganancias y perdidas.
                - **MXN:** pesos mexicanos. Cuando aparezca, indica que el valor
                  fue convertido usando el tipo de cambio USD/MXN simulado.
                - **pb (puntos base):** costo porcentual pequeño. `100 pb = 1%`,
                  `10 pb = 0.10%` y `1 pb = 0.01%`.
                - **Sesion:** un dia de mercado abierto. `63 sesiones` son
                  aproximadamente tres meses, no 63 dias de calendario.
                - **Trayectoria:** un futuro posible simulado. No representa
                  dinero, dias ni porcentaje; es un conteo de escenarios.
                - **Años:** cantidad de historia usada para calibrar el modelo.
                - **-1 a 1:** escala de correlacion. No es porcentaje.
                - **Sin unidad:** numero tecnico que no representa dinero ni
                  porcentaje, como Student-t df o la semilla.

                **Controles de Mercado**

                - **Precio inicial IVV (USD):** precio de una participacion al
                  comenzar la simulacion.
                - **Rendimiento anual base (%):** crecimiento promedio supuesto
                  para un año. La simulacion dura solo tres meses.
                - **Volatilidad anual (%):** intensidad de los movimientos del
                  precio. Un valor mayor significa mas incertidumbre.
                - **Inflacion y tasa de interes (% anual):** supuestos economicos
                  expresados para un año completo.
                - **Petroleo y USD/MXN (% a 3 meses):** cambio esperado durante
                  el mismo horizonte de la simulacion.

                **Controles de Estrategia**

                - **Caidas para comprar (%):** descensos desde el precio de
                  referencia que activan compras. `3, 6, 9` significa comprar
                  cuando la caida alcance 3%, 6% y 9%.
                - **Venta sobre costo promedio (%):** ganancia requerida para
                  vender despues de promediar todas las compras.
                - **Costo por operacion (pb):** comision o friccion aplicada a
                  cada compra y venta.

                **Trading agresivo**

                - **Capital (USD):** dinero inicial simulado.
                - **Caida, take-profit, stop-loss y trailing stop (%):** todos son
                  porcentajes respecto al precio de entrada o al maximo alcanzado.
                - **Maximo de operaciones:** cantidad de ciclos de compra y venta;
                  no es dinero ni porcentaje.
                """
            )

        with results_tab:
            st.markdown(
                """
                **Lee primero estas cuatro tarjetas**

                - **Retorno mediano:** el resultado que queda en medio de todos
                  los futuros simulados. No es una ganancia garantizada.
                - **Probabilidad de utilidad:** cuantas simulaciones terminaron
                  con ganancia. Por ejemplo, 62% significa 62 de cada 100.
                - **Probabilidad de activar compra:** cuantas veces el precio cayo
                  lo suficiente para que la estrategia comprara.
                - **Posicion abierta al final:** veces en que se compro, pero no se
                  alcanzo la meta de venta antes de terminar los tres meses.

                **Despues mira las graficas**

                - **Distribucion de resultados:** muestra resultados buenos y
                  malos. Cuanto mas extendida sea, mayor fue la incertidumbre.
                - **Abanico de precios:** la linea central es el resultado medio.
                  La franja muestra un rango amplio de futuros posibles.
                - **P5-P95:** 90 de cada 100 simulaciones quedaron dentro de ese
                  rango; 10 quedaron fuera.
                - **Precio final:** se muestra en USD por participacion de IVV.
                - **Retornos:** se muestran como porcentaje, no como dolares. Un
                  retorno de 5% sobre USD 1,000 equivale a USD 50 antes de costos.
                - **Capital y beneficio:** se muestran en USD.
                - **Probabilidades:** se muestran como porcentaje. 70% significa
                  que ocurrio en aproximadamente 70 de cada 100 simulaciones.
                - **Cantidad de salidas:** es un conteo de operaciones, no dinero.

                **Regla sencilla:** no mires solamente la posible ganancia. Revisa
                tambien cuantos casos perdieron dinero y cuantas posiciones
                quedaron abiertas.
                """
            )

        with glossary_tab:
            st.markdown(
                """
                - **IVV:** fondo que se compra y vende como una accion y busca
                  seguir a 500 empresas grandes de Estados Unidos.
                - **Trayectoria:** posible evolucion simulada del precio durante
                  aproximadamente tres meses.
                - **Monte Carlo:** metodo que repite miles de escenarios
                  posibles. No es una prediccion exacta.
                - **Retorno mediano:** resultado central; la mitad de las
                  trayectorias queda por encima y la otra mitad por debajo.
                - **P5-P95:** intervalo que contiene el 90% central de los
                  resultados simulados. No es una garantia.
                - **Volatilidad:** medida de la variacion esperada del precio. Un
                  valor mayor suele producir rangos de resultados mas amplios.
                  En esta app se expresa como porcentaje anual.
                - **Student-t:** distribucion usada para representar movimientos
                  extremos con mayor frecuencia que una distribucion normal.
                - **Student-t df:** parametro tecnico sin unidad. Valores menores
                  representan una mayor presencia de movimientos extremos.
                - **Drawdown o caida:** descenso desde un maximo previo que puede
                  activar una compra; se expresa como porcentaje.
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
                - **Correlacion:** relacion entre dos movimientos, medida de -1 a
                  1. Cerca de 1 se mueven juntos; cerca de -1, en sentido opuesto.
                - **MAPE:** error promedio expresado como porcentaje. Un valor
                  menor indica que el precio estimado estuvo mas cerca del real.
                - **R2:** proporcion explicada por el modelo, mostrada como
                  porcentaje. No representa rendimiento ni probabilidad.
                - **Posicion abierta:** compra que no alcanzo una regla de venta
                  antes de terminar el horizonte.
                """
            )


def display_basic_summary(results) -> None:
    profit = (results["strategy_return_usd"] > 0).mean()
    buy = (results["levels_triggered"] > 0).mean()
    open_position = results["position_open_at_horizon"].mean()
    median_return = results["strategy_return_usd"].median()

    st.subheader("Resultado explicado de forma sencilla")
    st.write(
        f"De cada 100 futuros simulados, aproximadamente **{profit * 100:.0f} "
        f"terminaron con ganancia** y **{buy * 100:.0f} activaron al menos una "
        f"compra**. En cerca de **{open_position * 100:.0f} de cada 100** quedo "
        "una inversion abierta al terminar los tres meses."
    )
    if median_return >= 0:
        st.success(
            f"El resultado central fue una ganancia de {median_return:.2%}. "
            "Recuerda: es el punto medio de la simulacion, no una promesa."
        )
    else:
        st.warning(
            f"El resultado central fue una perdida de {abs(median_return):.2%}. "
            "Esto indica que el escenario configurado merece cautela."
        )


def main() -> None:
    st.title("IVV Tactical Monte Carlo")
    st.caption(
        "Explora que podria pasar al comprar IVV durante una caida y vender "
        "despues de una recuperacion."
    )

    with st.sidebar:
        st.header("Tipo de usuario")
        app_mode = st.radio(
            "Elige como quieres usar la app",
            ("Basico", "Avanzado"),
            help="Basico simplifica las decisiones. Avanzado muestra todos los controles.",
        )

        try:
            default_price = latest_ivv_price()
        except Exception:
            default_price = 600.0
            st.warning("Se usa un precio provisional; revise la conexion.")

        st.header("Mercado")
        initial_price = st.number_input(
            "Precio actual aproximado de IVV (USD)",
            min_value=1.0,
            value=default_price,
            step=1.0,
            help="Precio desde el cual comenzara la simulacion.",
        )

        if app_mode == "Basico":
            profiles = {
                "Conservador": {
                    "return": 6.0,
                    "volatility": 15.0,
                    "levels": "5, 10, 15",
                    "take_profit": 5.0,
                },
                "Moderado": {
                    "return": 8.0,
                    "volatility": 18.0,
                    "levels": "3, 6, 9, 12",
                    "take_profit": 6.0,
                },
                "Agresivo": {
                    "return": 10.0,
                    "volatility": 25.0,
                    "levels": "2, 4, 6, 8",
                    "take_profit": 8.0,
                },
            }
            profile_name = st.selectbox(
                "Que nivel de riesgo quieres explorar?",
                tuple(profiles),
                index=1,
            )
            profile = profiles[profile_name]
            st.caption(
                {
                    "Conservador": "Espera caidas mayores antes de comprar.",
                    "Moderado": "Equilibra frecuencia de compra y riesgo.",
                    "Agresivo": "Compra antes y acepta movimientos mas fuertes.",
                }[profile_name]
            )
            expected_return = profile["return"]
            volatility = profile["volatility"]
            levels_text = profile["levels"]
            take_profit = profile["take_profit"]
            inflation = 3.0
            interest_rate = 4.0
            oil_change = 0.0
            fx_change = 0.0
            transaction_cost = 10.0
            paths = 10_000
            student_df = 5.0
            seed = 42
            use_calibration = False
            training_years = 5
            backtest_windows = 12
            aggressive_capital = 1_000.0
            aggressive_entry_label = "Esperar caida"
            aggressive_drawdown = 4.0
            aggressive_take_profit = 6.0
            aggressive_stop_loss = 5.0
            aggressive_trailing = 2.5
            aggressive_max_trades = 3
            run = st.button(
                "Ejecutar simulacion",
                type="primary",
                use_container_width=True,
            )
            run_backtest = False
        else:
            expected_return = st.slider(
                "Rendimiento anual base (%)",
                -20.0,
                30.0,
                8.0,
                0.5,
                help="Supuesto porcentual para un año completo.",
            )
            volatility = st.slider(
                "Volatilidad anual (%)",
                5.0,
                60.0,
                18.0,
                0.5,
                help="Variacion esperada anual. Mas volatilidad implica mas riesgo.",
            )
            inflation = st.slider(
                "Inflacion anual EE.UU. (%)",
                0.0,
                15.0,
                3.0,
                0.1,
                help="Inflacion supuesta para un año completo.",
            )
            interest_rate = st.slider(
                "Tasa de interes de referencia (% anual)",
                0.0,
                15.0,
                4.0,
                0.1,
            )
            oil_change = st.slider(
                "Cambio esperado petroleo (% a 3 meses)",
                -40.0,
                80.0,
                0.0,
                1.0,
            )
            fx_change = st.slider(
                "Cambio esperado USD/MXN (% a 3 meses)",
                -20.0,
                30.0,
                0.0,
                0.5,
            )

            st.header("Estrategia")
            levels_text = st.text_input(
                "Caidas que activan compras (%)",
                "3, 6, 9, 12",
                help="Porcentajes separados por comas, medidos desde el precio inicial.",
            )
            take_profit = st.slider(
                "Venta sobre costo promedio (%)", 1.0, 20.0, 6.0, 0.5
            )
            transaction_cost = st.number_input(
                "Costo por operacion (pb)",
                min_value=0.0,
                value=10.0,
                step=1.0,
                help="100 pb equivalen a 1%; 10 pb equivalen a 0.10%.",
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
                "Grados de libertad Student-t (sin unidad)",
                3.0,
                20.0,
                5.0,
                0.5,
            )
            seed = st.number_input(
                "Semilla aleatoria (sin unidad)",
                min_value=0,
                value=42,
                step=1,
                help="Permite repetir exactamente la misma simulacion.",
            )
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
                "Ventanas walk-forward (cantidad)", 4, 24, 12, 1
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
                "Caida para entrar (%)", 1.0, 20.0, 4.0, 0.5
            )
            aggressive_take_profit = st.slider(
                "Take-profit agresivo (%)", 1.0, 25.0, 6.0, 0.5
            )
            aggressive_stop_loss = st.slider(
                "Stop-loss agresivo (%)", 1.0, 20.0, 5.0, 0.5
            )
            aggressive_trailing = st.slider(
                "Trailing stop (%)", 0.5, 15.0, 2.5, 0.5
            )
            aggressive_max_trades = st.slider(
                "Maximo de operaciones (cantidad)", 1, 8, 3, 1
            )
            run = st.button(
                "Ejecutar simulacion", type="primary", use_container_width=True
            )
            run_backtest = st.button(
                "Ejecutar backtesting",
                use_container_width=True,
            )

    if app_mode == "Basico":
        st.info(
            "El modo Basico usa valores preparados para que puedas concentrarte "
            "en entender los resultados. Empieza con el perfil Moderado."
        )
    else:
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
        if app_mode == "Basico":
            st.subheader("Que hace esta pagina?")
            st.write(
                "Imagina 10,000 futuros posibles para los proximos tres meses. "
                "En cada futuro, la pagina aplica las mismas reglas de compra y "
                "venta. Al final cuenta cuantas veces hubo ganancia, perdida o "
                "una compra que todavia no se habia vendido."
            )
        else:
            st.subheader("Que evalua esta demo")
            st.write(
                "El motor genera 63 sesiones con volatilidad tipo GARCH, "
                "innovaciones Student-t y saltos distintos por escenario. En "
                "cada trayectoria aplica las compras configuradas y compara su "
                "resultado con comprar IVV hoy."
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
    if calibration is not None and app_mode == "Avanzado":
        display_calibration(calibration)
    metric_labels = (
        (
            "Resultado central",
            "Casos con ganancia",
            "Casos donde se compro",
            "Compra sin vender al final",
        )
        if app_mode == "Basico"
        else (
            "Retorno mediano estrategia (%)",
            "Probabilidad de utilidad (%)",
            "Probabilidad de activar compra (%)",
            "Posicion abierta al final (%)",
        )
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        metric_labels[0],
        percentage(results["strategy_return_usd"].median()),
    )
    col2.metric(
        metric_labels[1],
        percentage((results["strategy_return_usd"] > 0).mean()),
    )
    col3.metric(
        metric_labels[2],
        percentage((results["levels_triggered"] > 0).mean()),
    )
    col4.metric(
        metric_labels[3],
        percentage(results["position_open_at_horizon"].mean()),
    )

    if app_mode == "Basico":
        display_basic_summary(results)
    else:
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

    if app_mode == "Avanzado":
        st.subheader("Percentiles")
        percentiles = output["percentiles"].copy()
        for column in percentiles.columns:
            if column != "final_price":
                percentiles[column] = percentiles[column].map(percentage)
            else:
                percentiles[column] = percentiles[column].map(
                    lambda value: f"${value:,.2f}"
                )
        percentiles = percentiles.rename(
            columns={
                "strategy_return_usd": "Retorno estrategia en USD (%)",
                "strategy_return_mxn": "Retorno estrategia en MXN (%)",
                "buy_hold_return_usd": "Retorno comprar y mantener (%)",
                "final_price": "Precio final IVV (USD)",
            }
        )
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
        scenarios = scenarios.rename(
            columns={
                "scenario": "Escenario",
                "weight": "Peso del escenario (%)",
                "paths": "Trayectorias (cantidad)",
                "median_strategy_usd": "Retorno mediano en USD (%)",
                "probability_profit": "Probabilidad de utilidad (%)",
                "probability_buy": "Probabilidad de compra (%)",
                "probability_sale": "Probabilidad de venta (%)",
                "probability_open": "Posicion abierta al final (%)",
                "median_final_price": "Precio final mediano (USD)",
            }
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
