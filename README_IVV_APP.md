# IVV Tactical Monte Carlo

Aplicacion de investigacion para evaluar una estrategia de compras escalonadas
durante caidas de IVV y venta durante la recuperacion.

## Modelo

- Horizonte de 63 sesiones, aproximadamente tres meses.
- Escenarios: 15% adverso, 70% central y 15% favorable.
- Innovaciones Student-t para colas pesadas.
- Volatilidad recursiva tipo GARCH.
- Saltos de precio condicionados al escenario.
- Supuestos de inflacion, tasas, petroleo y USD/MXN.
- Resultado de la estrategia en USD y MXN.
- Comparacion contra comprar y mantener IVV.
- Calibracion historica de retorno, volatilidad, colas y escenarios.
- Backtesting walk-forward sin utilizar informacion futura.
- Seccion independiente de trading agresivo con capital configurable.
- Entrada inmediata o espera de una caida desde el maximo reciente.
- Take-profit, stop-loss, trailing stop y multiples ciclos de compra-venta.
- Graficos Plotly interactivos con zoom, hover y descarga.

La app permite alternar entre supuestos manuales y parametros historicamente
calibrados. La inflacion permanece como supuesto de baja frecuencia.

## Instalacion

```bash
python -m pip install -r requirements-ivv-app.txt
```

## Ejecucion

```bash
streamlit run app_ivv_trading.py
```

## Publicacion

La ruta recomendada para una primera version publica es Streamlit Community
Cloud. Consulta `DEPLOY_STREAMLIT.md` para subirla desde GitHub y obtener una
URL compartible.

## Calibracion y backtesting

La calibracion usa IVV, WTI, Treasury a 10 anos y USD/MXN. Estima:

- Retorno y volatilidad anualizados.
- Grados de libertad de Student-t a partir de curtosis.
- Sensibilidades macro mediante regresion ridge.
- Correlacion entre IVV y USD/MXN.
- Regimen adverso, central y favorable mediante retornos moviles.
- Frecuencia, media y dispersion de saltos.

El walk-forward calibra cada origen exclusivamente con observaciones anteriores,
simula los siguientes 63 dias y compara P5-P95 con el resultado observado.

## Trading agresivo

La seccion agresiva permite simular USD 1,000 o cualquier otro capital mediante:

- Compra inmediata o entrada al alcanzar un drawdown seleccionado.
- Inversion del capital completo en cada ciclo.
- Salida por take-profit, stop-loss o trailing stop.
- Reentrada despues de un periodo de enfriamiento.
- Comparacion entre el beneficio realizado y el maximo teorico posterior.

El dia de maximo teorico se conoce unicamente de forma retrospectiva. No se usa
como regla de operacion ni se presenta como una prediccion exacta.

## Siguiente etapa profesional

1. Sustituir Yahoo Finance por fuentes oficiales y almacenamiento versionado.
2. Incorporar CPI y expectativas de inflacion respetando su frecuencia mensual.
3. Estimar GARCH con maxima verosimilitud y validar residuos.
4. Etiquetar eventos geopoliticos historicos y calibrar saltos condicionados.
5. Incorporar impuestos, deslizamiento y reglas de administracion de riesgo.
