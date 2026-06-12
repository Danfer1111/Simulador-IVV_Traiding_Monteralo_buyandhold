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
    MarketAssumptions,
    SimulationConfig,
    TradingStrategy,
    run_simulation,
)
from ivv_paper_trading import (
    PaperTradingConfig,
    PaperTradingStrategy,
    run_paper_trading,
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


def select_operational_path(prices: np.ndarray, path_label: str) -> np.ndarray:
    target_quantile = {
        "Adversa (P10)": 0.10,
        "Mediana (P50)": 0.50,
        "Favorable (P90)": 0.90,
    }[path_label]
    final_prices = prices[:, -1]
    target = np.quantile(final_prices, target_quantile)
    path_index = int(np.argmin(np.abs(final_prices - target)))
    return prices[path_index].copy()


def display_paper_trading(
    prices: np.ndarray,
    strategy: PaperTradingStrategy,
    config: PaperTradingConfig,
    path_label: str,
) -> None:
    operational_prices = select_operational_path(prices, path_label)
    output = run_paper_trading(operational_prices, strategy, config)
    sessions = output["sessions"]
    events = output["events"]
    summary = output["summary"]
    theoretical = output["theoretical"]

    st.divider()
    st.header("Paper trading operativo")
    st.warning(
        "Entorno 100% simulado: no solicita credenciales, no se conecta a un "
        "broker y no puede enviar ordenes reales."
    )
    st.caption(
        f"Trayectoria utilizada: {path_label}. Cada orden se evalua con apertura, "
        "maximo, minimo, bid, ask, vigencia, costos y deslizamiento simulados."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capital inicial (USD)", f"USD {summary['initial_capital']:,.2f}")
    col2.metric(
        "Patrimonio final (USD)",
        f"USD {summary['final_equity']:,.2f}",
        percentage(summary["return"]),
    )
    col3.metric("Maxima caida de patrimonio (%)", percentage(summary["max_drawdown"]))
    col4.metric("Ordenes ejecutadas (cantidad)", summary["orders_filled"])

    status1, status2, status3, status4 = st.columns(4)
    status1.metric("Ordenes creadas", summary["orders_created"])
    status2.metric("Ordenes vencidas", summary["orders_expired"])
    status3.metric("Ordenes rechazadas", summary["orders_rejected"])
    status4.metric(
        "Control de riesgo",
        "DETENIDO" if summary["risk_halted"] else "Activo",
    )

    st.subheader("Resultado ejecutable frente al maximo teorico")
    st.caption(
        "El maximo teorico usa informacion futura para elegir la mejor compra y "
        "la mejor venta posterior. Incluye los mismos costos, spread, "
        "deslizamiento y limite de exposicion, pero no es ejecutable en tiempo real."
    )
    comparison1, comparison2, comparison3, comparison4 = st.columns(4)
    comparison1.metric(
        "Beneficio ejecutable (USD)",
        f"USD {summary['actual_profit']:,.2f}",
    )
    comparison2.metric(
        "Beneficio teorico perfecto (USD)",
        f"USD {summary['theoretical_profit']:,.2f}",
    )
    comparison3.metric(
        "Movimiento capturado (%)",
        percentage(summary["capture_ratio"]),
    )
    if theoretical["trade_available"]:
        comparison4.metric(
            "Compra / venta teoricas",
            f"Sesion {theoretical['entry_session']} / "
            f"{theoretical['exit_session']}",
        )
        st.info(
            f"Con conocimiento futuro perfecto se compraria cerca de "
            f"**USD {theoretical['entry_fill_price']:,.2f}** en la sesion "
            f"**{theoretical['entry_session']}** y se venderia cerca de "
            f"**USD {theoretical['exit_fill_price']:,.2f}** en la sesion "
            f"**{theoretical['exit_session']}**."
        )
    else:
        comparison4.metric("Compra / venta teoricas", "Sin operacion rentable")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=sessions["session"],
                open=sessions["open"],
                high=sessions["high"],
                low=sessions["low"],
                close=sessions["close"],
                name="IVV simulado",
            )
        )
        if not events.empty:
            purchases = events.loc[
                (events["side"] == "Compra") & (events["status"] == "Ejecutada")
            ]
            sales = events.loc[
                (events["side"] == "Venta") & (events["status"] == "Ejecutada")
            ]
            fig.add_trace(
                go.Scatter(
                    x=purchases["session"],
                    y=purchases["fill_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", size=13, color="#2E7D32"),
                    name="Compra ejecutada",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sales["session"],
                    y=sales["fill_price"],
                    mode="markers",
                    marker=dict(symbol="triangle-down", size=13, color="#C62828"),
                    name="Venta ejecutada",
                )
            )
        if theoretical["trade_available"]:
            fig.add_trace(
                go.Scatter(
                    x=[
                        theoretical["entry_session"],
                        theoretical["exit_session"],
                    ],
                    y=[
                        theoretical["entry_market_low"],
                        theoretical["exit_market_high"],
                    ],
                    mode="markers+lines",
                    marker=dict(
                        symbol=["star", "star"],
                        size=16,
                        color=["#1565C0", "#8E24AA"],
                    ),
                    line=dict(color="rgba(80,80,80,0.45)", dash="dot"),
                    text=["Compra teorica perfecta", "Venta teorica perfecta"],
                    hovertemplate=(
                        "%{text}<br>Sesion %{x}<br>Precio: USD %{y:,.2f}"
                        "<extra></extra>"
                    ),
                    name="Operacion teorica perfecta",
                )
            )
        fig.update_layout(
            title="Precio y ejecuciones simuladas",
            xaxis_title="Sesion",
            yaxis_title="IVV (USD)",
            xaxis_rangeslider_visible=False,
        )
        show_plotly(fig)

    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sessions["session"],
                y=sessions["equity"],
                mode="lines",
                line=dict(color="#174A8B", width=3),
                name="Patrimonio",
                hovertemplate="Patrimonio: USD %{y:,.2f}<extra></extra>",
            )
        )
        loss_limit = config.initial_capital_usd * (1 - config.max_portfolio_loss)
        fig.add_hline(
            y=loss_limit,
            line_color="#C62828",
            line_dash="dash",
            annotation_text="Limite de perdida",
        )
        fig.update_layout(
            title="Patrimonio y limite de riesgo",
            xaxis_title="Sesion",
            yaxis_title="Patrimonio (USD)",
        )
        show_plotly(fig)

    st.subheader("Bitacora de ordenes y controles")
    if events.empty:
        st.info("La trayectoria no genero señales ni ordenes.")
    else:
        event_table = events.copy()
        event_table = event_table.rename(
            columns={
                "session": "Sesion",
                "event": "Evento",
                "side": "Lado",
                "status": "Estado",
                "reason": "Motivo",
                "limit_price": "Precio limite (USD)",
                "fill_price": "Precio ejecutado (USD)",
                "quantity": "Participaciones",
                "cash": "Efectivo (USD)",
                "position_shares": "Posicion (participaciones)",
                "equity": "Patrimonio (USD)",
            }
        )
        st.dataframe(
            event_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Precio limite (USD)": st.column_config.NumberColumn(format="%.2f"),
                "Precio ejecutado (USD)": st.column_config.NumberColumn(format="%.2f"),
                "Participaciones": st.column_config.NumberColumn(format="%.4f"),
                "Efectivo (USD)": st.column_config.NumberColumn(format="%.2f"),
                "Posicion (participaciones)": st.column_config.NumberColumn(
                    format="%.4f"
                ),
                "Patrimonio (USD)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    with st.expander("Ver estado de cada sesion"):
        session_table = sessions.rename(
            columns={
                "session": "Sesion",
                "open": "Apertura (USD)",
                "high": "Maximo (USD)",
                "low": "Minimo (USD)",
                "close": "Cierre (USD)",
                "bid": "Bid (USD)",
                "ask": "Ask (USD)",
                "drawdown": "Caida desde maximo (%)",
                "cash": "Efectivo (USD)",
                "position_shares": "Posicion (participaciones)",
                "average_cost": "Costo promedio (USD)",
                "equity": "Patrimonio (USD)",
                "return": "Retorno (%)",
                "pending_order": "Orden pendiente",
                "risk_halted": "Riesgo detenido",
            }
        ).copy()
        session_table["Caida desde maximo (%)"] = session_table[
            "Caida desde maximo (%)"
        ].map(percentage)
        session_table["Retorno (%)"] = session_table["Retorno (%)"].map(percentage)
        st.dataframe(session_table, use_container_width=True, hide_index=True)


def display_user_guide() -> None:
    with st.expander("Manual de uso y glosario", expanded=False):
        start_tab, advanced_tab, paper_tab, results_tab, glossary_tab = st.tabs(
            (
                "Empieza aqui",
                "Modo avanzado",
                "Paper trading",
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
                los supuestos, ejecutar pruebas historicas y configurar paper
                trading operativo.
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
                - **Precio de IVV:** costo de una participacion. La app usa un
                  unico precio inicial para la estrategia normal, comprar y
                  mantener y paper trading.
                - **Capital:** dinero disponible para invertir. No es el precio
                  de IVV. Por ejemplo, puedes tener USD 1,000 de capital aunque
                  una participacion de IVV cueste una cantidad diferente.
                - **Capital final:** valor del efectivo y las participaciones al
                  terminar la simulacion.
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

                **Paper trading**

                - **Capital (USD):** dinero inicial simulado.
                - **Caida, take-profit, stop-loss y trailing stop (%):** todos son
                  porcentajes respecto al precio de entrada o al maximo alcanzado.
                - **Exposicion maxima (%):** parte del capital que puede quedar
                  invertida al mismo tiempo.
                - **Perdida maxima (%):** nivel que detiene nuevas operaciones.
                """
            )

        with paper_tab:
            st.markdown(
                """
                **Que hace esta seccion**

                El paper trading toma una sola trayectoria del Monte Carlo y la
                recorre como si fueran sesiones de mercado. No usa dinero real.

                1. Selecciona una trayectoria adversa, mediana o favorable.
                2. Cuando se alcanza una caida configurada, crea una orden limite.
                3. La orden solo se ejecuta si el minimo de una sesion posterior
                   alcanza el precio limite.
                4. Si no se ejecuta dentro de su vigencia, la orden vence.
                5. Las ventas siguen take-profit, stop-loss y trailing stop.
                6. Los controles pueden rechazar ordenes o detener el sistema.

                **Elementos realistas incluidos**

                - **Bid/ask:** diferencia simulada entre precio comprador y vendedor.
                - **Deslizamiento:** diferencia posible entre el precio esperado y
                  el precio ejecutado.
                - **Comision:** costo aplicado al comprar y vender.
                - **Vigencia:** numero de sesiones que una orden puede esperar.
                - **Exposicion maxima:** porcentaje maximo del capital invertido.
                - **Perdida maxima:** nivel que cancela ordenes y detiene operaciones.
                - **Bitacora:** registro de señales, ordenes, ejecuciones, rechazos
                  y controles de riesgo.
                - **Maximo teorico:** mejor compra y venta posterior identificadas
                  despues de conocer toda la trayectoria.
                - **Movimiento capturado:** beneficio ejecutable dividido entre
                  el beneficio teorico perfecto. Por ejemplo, 60% significa que la
                  estrategia capturo 60 de cada 100 dolares del beneficio ideal.

                **Importante:** los maximos y minimos intradia tambien son
                simulados. Esta seccion sirve para probar el proceso operativo,
                no para demostrar que una orden real se habria ejecutado. La
                operacion teorica perfecta nunca debe interpretarse como señal.
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
                - **Capital disponible:** dinero usado para comprar. No debe
                  confundirse con el precio de una participacion de IVV.
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
                - **Ganancia teorica perfecta:** beneficio retrospectivo de comprar
                  en el mejor minimo y vender en el mejor maximo posterior.
                - **Captura del movimiento:** parte de esa ganancia teorica que
                  consiguio la estrategia ejecutable.
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
            paper_enabled = False
            paper_path_label = "Mediana (P50)"
            paper_capital = 10_000.0
            paper_max_exposure = 80.0
            paper_max_loss = 10.0
            paper_order_expiry = 2
            paper_limit_offset = 5.0
            paper_spread = 4.0
            paper_slippage = 2.0
            paper_stop_loss = 5.0
            paper_trailing_stop = 3.0
            paper_fractional = True
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
            st.header("Paper trading operativo")
            paper_enabled = st.checkbox(
                "Simular operacion con reglas reales",
                value=True,
                help="No conecta un broker ni envia ordenes reales.",
            )
            paper_path_label = st.selectbox(
                "Trayectoria operativa",
                ("Adversa (P10)", "Mediana (P50)", "Favorable (P90)"),
                index=1,
                help="Selecciona un futuro representativo para recorrer sesion por sesion.",
            )
            paper_capital = st.number_input(
                "Capital simulado de paper trading (USD)",
                min_value=100.0,
                value=10_000.0,
                step=500.0,
            )
            paper_max_exposure = st.slider(
                "Exposicion maxima del capital (%)", 10.0, 100.0, 80.0, 5.0
            )
            paper_max_loss = st.slider(
                "Perdida maxima antes de detener (%)", 1.0, 30.0, 10.0, 1.0
            )
            paper_order_expiry = st.slider(
                "Vigencia de orden limite (sesiones)", 1, 10, 2, 1
            )
            paper_limit_offset = st.slider(
                "Precio limite debajo del ask (pb)", 0.0, 100.0, 5.0, 1.0
            )
            paper_spread = st.slider(
                "Diferencial bid/ask simulado (pb)", 0.0, 50.0, 4.0, 1.0
            )
            paper_slippage = st.slider(
                "Deslizamiento simulado (pb)", 0.0, 50.0, 2.0, 1.0
            )
            paper_stop_loss = st.slider(
                "Stop-loss de paper trading (%)", 1.0, 20.0, 5.0, 0.5
            )
            paper_trailing_stop = st.slider(
                "Trailing stop de paper trading (%)", 0.5, 15.0, 3.0, 0.5
            )
            paper_fractional = st.checkbox(
                "Permitir participaciones fraccionadas",
                value=True,
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
    simulation_initial_price = float(output["prices"][0, 0])
    st.metric(
        "Precio unico de IVV usado en toda la simulacion",
        f"USD {simulation_initial_price:,.2f}",
    )
    st.caption(
        "Este mismo precio se usa para comprar y mantener, compras escalonadas "
        "y paper trading."
    )
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

        if paper_enabled:
            paper_allocation = paper_max_exposure / 100 / len(levels)
            paper_strategy = PaperTradingStrategy(
                drawdown_levels=levels,
                allocations=tuple(paper_allocation for _ in levels),
                take_profit=take_profit / 100,
                stop_loss=paper_stop_loss / 100,
                trailing_stop=paper_trailing_stop / 100,
            )
            paper_config = PaperTradingConfig(
                initial_capital_usd=paper_capital,
                spread_bps=paper_spread,
                slippage_bps=paper_slippage,
                transaction_cost_bps=transaction_cost,
                limit_offset_bps=paper_limit_offset,
                order_expiry_sessions=paper_order_expiry,
                max_exposure=paper_max_exposure / 100,
                max_portfolio_loss=paper_max_loss / 100,
                allow_fractional_shares=paper_fractional,
            )
            display_paper_trading(
                output["prices"],
                paper_strategy,
                paper_config,
                paper_path_label,
            )

    st.warning(
        "Demo de investigacion, no recomendacion de inversion. Las sensibilidades "
        "macro y los shocks geopoliticos aun deben calibrarse y validarse mediante "
        "backtesting walk-forward antes de usar el resultado para operar."
    )


if __name__ == "__main__":
    main()
